mod gpu;
mod term;
mod window;

use anyhow::{Context, Result};
use crossterm::event::{self, Event, KeyCode, KeyModifiers, MouseButton, MouseEventKind};
use notify::{RecursiveMode, Watcher};
use std::path::PathBuf;
use std::sync::mpsc;
use std::time::{Duration, Instant};

struct Args {
    file: PathBuf,
    fps: u32,
    scale: f32,
    kitty: bool,
    window: bool,
}

fn parse_args() -> Result<Args> {
    let mut file = None;
    let mut fps = 30;
    let mut scale = 1.0;
    let mut kitty = false;
    let mut window = false;
    let mut it = std::env::args().skip(1);
    while let Some(a) = it.next() {
        match a.as_str() {
            "--fps" => fps = it.next().context("--fps needs a value")?.parse()?,
            "--scale" => scale = it.next().context("--scale needs a value")?.parse()?,
            "--kitty" => kitty = true,
            "--window" | "-w" => window = true,
            "-h" | "--help" => {
                println!(
                    "usage: shadertoy <shader.glsl> [--window] [--fps N] [--scale F] [--kitty]\n\n\
                     Live-reloading Shadertoy-style GLSL renderer for the terminal.\n\
                     --window    open a native window rendering at its real pixel size
                                 (ignores --fps/--scale/--kitty; vsync-paced)
                     --fps N     target frame rate (default 30)\n\
                     --scale F   resolution multiplier for half-block mode (default 1.0)\n\
                     --kitty     use the Kitty graphics protocol instead of half-blocks\n\
                                 (needs a kitty-compatible terminal; under tmux also\n\
                                 `set -g allow-passthrough on`)\n\n\
                     keys: q quit | space pause | r restart time"
                );
                std::process::exit(0);
            }
            _ if file.is_none() => file = Some(PathBuf::from(a)),
            _ => anyhow::bail!("unexpected argument: {a}"),
        }
    }
    Ok(Args {
        file: file.context("usage: shadertoy <shader.glsl> [--window] [--fps N] [--scale F] [--kitty]")?,
        fps: fps.max(1),
        scale,
        kitty,
        window,
    })
}

fn main() -> Result<()> {
    let args = parse_args()?;
    if args.window {
        return window::run(args.file);
    }
    let mut gpu = gpu::Gpu::new()?;
    let adapter_name = gpu.adapter_name.clone();

    // File watching: watch the parent directory, because many editors save by
    // writing a temp file and renaming it, which kills a watch on the file itself.
    let abs = std::fs::canonicalize(&args.file)
        .with_context(|| format!("cannot open {}", args.file.display()))?;
    let dir = abs.parent().unwrap().to_path_buf();
    let (tx, rx) = mpsc::channel();
    let target = abs.clone();
    let mut watcher = notify::recommended_watcher(move |res: notify::Result<notify::Event>| {
        if let Ok(ev) = res {
            if ev.paths.iter().any(|p| p.file_name() == target.file_name()) {
                let _ = tx.send(());
            }
        }
    })?;
    watcher.watch(&dir, RecursiveMode::NonRecursive)?;

    let mut status = String::new();
    let mut status_is_err = false;
    let load = |gpu: &mut gpu::Gpu, status: &mut String, is_err: &mut bool| {
        match std::fs::read_to_string(&abs) {
            Ok(src) => match gpu.set_shader(&src) {
                Ok(()) => {
                    *status = "ok".into();
                    *is_err = false;
                }
                Err(e) => {
                    *status = e;
                    *is_err = true;
                }
            },
            Err(e) => {
                *status = format!("read error: {e}");
                *is_err = true;
            }
        }
    };
    load(&mut gpu, &mut status, &mut status_is_err);

    let mut term = term::Term::enter(args.kitty)?;
    let frame_dt = Duration::from_secs_f64(1.0 / args.fps as f64);
    let start = Instant::now();
    let mut paused_at: Option<Instant> = None;
    let mut time_offset = Duration::ZERO; // time spent paused
    let mut last_frame = Instant::now();
    let mut frame_no: i32 = 0;
    let mut mouse = [0.0f32; 4];
    let mut fps_ema = args.fps as f32;

    'main: loop {
        let deadline = Instant::now() + frame_dt;

        // Reload (debounced: drain everything queued, then reload once).
        if rx.try_recv().is_ok() {
            while rx.try_recv().is_ok() {}
            std::thread::sleep(Duration::from_millis(20)); // let the write finish
            while rx.try_recv().is_ok() {}
            load(&mut gpu, &mut status, &mut status_is_err);
        }

        // Input
        loop {
            let left = deadline.saturating_duration_since(Instant::now());
            if !event::poll(left)? {
                break;
            }
            match event::read()? {
                Event::Key(k) if k.kind != event::KeyEventKind::Release => match k.code {
                    KeyCode::Char('q') | KeyCode::Esc => break 'main,
                    KeyCode::Char('c') if k.modifiers.contains(KeyModifiers::CONTROL) => {
                        break 'main
                    }
                    KeyCode::Char(' ') => match paused_at.take() {
                        Some(t) => time_offset += t.elapsed(),
                        None => paused_at = Some(Instant::now()),
                    },
                    KeyCode::Char('r') => {
                        time_offset = start.elapsed();
                        if paused_at.is_some() {
                            paused_at = Some(Instant::now());
                            time_offset = paused_at.unwrap() - start;
                        }
                        frame_no = 0;
                    }
                    _ => {}
                },
                Event::Mouse(m) => {
                    let (w, h) = term.pixel_size(args.scale);
                    let (cw, ch) = term.cell_size_in_pixels(w, h);
                    let x = m.column as f32 * cw + cw / 2.0;
                    let y = h as f32 - (m.row as f32 * ch + ch / 2.0);
                    match m.kind {
                        MouseEventKind::Down(MouseButton::Left) => {
                            mouse = [x, y, x, y];
                        }
                        MouseEventKind::Drag(MouseButton::Left) => {
                            mouse[0] = x;
                            mouse[1] = y;
                        }
                        MouseEventKind::Up(MouseButton::Left) => {
                            // Shadertoy: sign of zw encodes button state.
                            mouse[2] = -mouse[2].abs();
                            mouse[3] = -mouse[3].abs();
                        }
                        _ => {}
                    }
                }
                _ => {}
            }
        }

        // Frame
        let now = Instant::now();
        let delta = now.duration_since(last_frame).as_secs_f32();
        last_frame = now;
        if delta > 0.0 {
            fps_ema = fps_ema * 0.9 + (1.0 / delta) * 0.1;
        }
        let t = match paused_at {
            Some(p) => p - start - time_offset,
            None => start.elapsed() - time_offset,
        }
        .as_secs_f32();

        let (w, h) = term.pixel_size(args.scale);
        let frame = gpu.render(w, h, t, delta, frame_no, mouse)?;
        if paused_at.is_none() {
            frame_no = frame_no.wrapping_add(1);
        }

        let info = format!(
            "{} | {}x{} | {:.0} fps | {} | {}",
            args.file.file_name().unwrap().to_string_lossy(),
            w,
            h,
            fps_ema,
            adapter_name,
            if paused_at.is_some() { "paused" } else { "" }
        );
        term.draw(&frame, w, h, &info, &status, status_is_err)?;
    }
    Ok(())
}
