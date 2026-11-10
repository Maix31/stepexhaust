use anyhow::Result;
use base64::Engine;
use crossterm::{
    cursor, event, execute,
    terminal::{self, EnterAlternateScreen, LeaveAlternateScreen},
};
use std::io::{Stdout, Write};

pub struct Term {
    out: Stdout,
    kitty: bool,
    buf: Vec<u8>,
}

impl Term {
    pub fn enter(kitty: bool) -> Result<Self> {
        terminal::enable_raw_mode()?;
        let mut out = std::io::stdout();
        execute!(
            out,
            EnterAlternateScreen,
            cursor::Hide,
            event::EnableMouseCapture,
            terminal::Clear(terminal::ClearType::All)
        )?;
        // Restore the terminal even if we panic.
        let prev = std::panic::take_hook();
        std::panic::set_hook(Box::new(move |info| {
            restore();
            prev(info);
        }));
        Ok(Self { out, kitty, buf: Vec::with_capacity(1 << 20) })
    }

    /// Render resolution in pixels.
    pub fn pixel_size(&self, scale: f32) -> (u32, u32) {
        let (cols, rows) = terminal::size().unwrap_or((80, 24));
        let rows = rows.saturating_sub(1).max(1); // last row = status line
        if self.kitty {
            if let Ok(ws) = terminal::window_size() {
                if ws.width > 0 && ws.height > 0 {
                    let cw = ws.width as f32 / ws.columns as f32;
                    let ch = ws.height as f32 / ws.rows as f32;
                    let w = (cols as f32 * cw * scale) as u32;
                    let h = (rows as f32 * ch * scale) as u32;
                    return (w.clamp(1, 1920), h.clamp(1, 1080));
                }
            }
            return (cols as u32 * 8, rows as u32 * 16);
        }
        (
            ((cols as f32 * scale) as u32).max(1),
            ((rows as f32 * 2.0 * scale) as u32).max(1),
        )
    }

    /// Size of one terminal cell expressed in render pixels (for mouse mapping).
    pub fn cell_size_in_pixels(&self, w: u32, h: u32) -> (f32, f32) {
        let (cols, rows) = terminal::size().unwrap_or((80, 24));
        let rows = rows.saturating_sub(1).max(1);
        (w as f32 / cols as f32, h as f32 / rows as f32)
    }

    pub fn draw(
        &mut self,
        rgba: &[u8],
        w: u32,
        h: u32,
        info: &str,
        status: &str,
        is_err: bool,
    ) -> Result<()> {
        let (cols, rows) = terminal::size().unwrap_or((80, 24));
        self.buf.clear();
        // Synchronized update: avoids tearing on terminals that support it.
        self.buf.extend_from_slice(b"\x1b[?2026h\x1b[2;1H"); // image starts below the top status row
        if self.kitty {
            self.kitty_image(rgba, w, h, cols, rows.saturating_sub(1).max(1));
        } else {
            self.half_blocks(rgba, w, h, cols, rows.saturating_sub(1).max(1));
        }
        self.status_line(info, status, is_err, cols);
        self.buf.extend_from_slice(b"\x1b[?2026l");
        self.out.write_all(&self.buf)?;
        self.out.flush()?;
        Ok(())
    }

    fn half_blocks(&mut self, rgba: &[u8], w: u32, h: u32, cols: u16, rows: u16) {
        let px = |x: u32, y: u32| -> [u8; 3] {
            let x = x.min(w - 1);
            let y = y.min(h - 1);
            let i = ((y * w + x) * 4) as usize;
            [rgba[i], rgba[i + 1], rgba[i + 2]]
        };
        let mut last: Option<([u8; 3], [u8; 3])> = None;
        for row in 0..rows as u32 {
            // Move to the start of each row explicitly: robust against wrapping.
            self.buf.extend_from_slice(format!("\x1b[{};1H", row + 2).as_bytes());
            for col in 0..cols as u32 {
                // Nearest-neighbour map from cell grid onto the render target.
                let x = col * w / cols as u32;
                let ytop = (row * 2) * h / (rows as u32 * 2);
                let ybot = (row * 2 + 1) * h / (rows as u32 * 2);
                let (top, bot) = (px(x, ytop), px(x, ybot));
                if last != Some((top, bot)) {
                    let _ = write!(
                        self.buf,
                        "\x1b[38;2;{};{};{};48;2;{};{};{}m",
                        top[0], top[1], top[2], bot[0], bot[1], bot[2]
                    );
                    last = Some((top, bot));
                }
                self.buf.extend_from_slice("▀".as_bytes());
            }
        }
        self.buf.extend_from_slice(b"\x1b[0m");
    }

    fn kitty_image(&mut self, rgba: &[u8], w: u32, h: u32, cols: u16, rows: u16) {
        // Raw RGBA (f=32), id 1 so each frame replaces the previous one.
        let b64 = base64::engine::general_purpose::STANDARD.encode(rgba);
        let tmux = std::env::var_os("TMUX").is_some();
        let chunks: Vec<&[u8]> = b64.as_bytes().chunks(4096).collect();
        for (i, chunk) in chunks.iter().enumerate() {
            let more = if i + 1 < chunks.len() { 1 } else { 0 };
            let mut seq = Vec::new();
            if i == 0 {
                seq.extend_from_slice(
                    format!("\x1b_Ga=T,f=32,i=1,q=2,s={w},v={h},c={cols},r={rows},C=1,m={more};")
                        .as_bytes(),
                );
            } else {
                seq.extend_from_slice(format!("\x1b_Gm={more};").as_bytes());
            }
            seq.extend_from_slice(chunk);
            seq.extend_from_slice(b"\x1b\\");
            if tmux {
                // tmux passthrough: wrap in DCS and double every ESC.
                self.buf.extend_from_slice(b"\x1bPtmux;");
                for &b in &seq {
                    if b == 0x1b {
                        self.buf.push(0x1b);
                    }
                    self.buf.push(b);
                }
                self.buf.extend_from_slice(b"\x1b\\");
            } else {
                self.buf.extend_from_slice(&seq);
            }
        }
    }

    fn status_line(&mut self, info: &str, status: &str, is_err: bool, cols: u16) {
        // Only the first line of an error, so the layout never jumps around.
        let msg = if is_err {
            format!("ERROR {}", status.lines().next().unwrap_or(""))
        } else {
            info.to_string()
        };
        let mut s: String = msg.chars().take(cols as usize).collect();
        let pad = (cols as usize).saturating_sub(s.chars().count());
        s.push_str(&" ".repeat(pad));
        let color = if is_err { "\x1b[97;41m" } else { "\x1b[30;47m" };
        let _ = write!(self.buf, "\x1b[1;1H{color}{s}\x1b[0m");
    }
}

fn restore() {
    let mut out = std::io::stdout();
    let _ = execute!(
        out,
        event::DisableMouseCapture,
        cursor::Show,
        LeaveAlternateScreen
    );
    let _ = terminal::disable_raw_mode();
}

impl Drop for Term {
    fn drop(&mut self) {
        if self.kitty {
            // delete our image
            let _ = self.out.write_all(b"\x1b_Ga=d,d=I,i=1,q=2\x1b\\");
        }
        restore();
    }
}
