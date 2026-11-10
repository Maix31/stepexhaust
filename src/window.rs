//! `--window` mode: render straight to a native window at its real pixel size.

use crate::gpu::Gpu;
use anyhow::{anyhow, Context, Result};
use notify::{RecursiveMode, Watcher};
use std::path::PathBuf;
use std::sync::{mpsc, Arc};
use std::time::{Duration, Instant};
use winit::application::ApplicationHandler;
use winit::dpi::LogicalSize;
use winit::event::{ElementState, MouseButton, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::{Key, NamedKey};
use winit::window::{Window, WindowId};

struct State {
    window: Arc<Window>,
    surface: wgpu::Surface<'static>,
    config: wgpu::SurfaceConfiguration,
    gpu: Gpu,
    status: String,
    status_is_err: bool,
    start: Instant,
    paused_at: Option<Instant>,
    time_offset: Duration, // time spent paused
    last_frame: Instant,
    frame_no: i32,
    mouse: [f32; 4],
    cursor: [f32; 2],
    fps_ema: f32,
    render_ms_ema: f32,
    last_title: Instant,
    last_overlay: Option<Instant>,
    last_err: Option<String>,
}

struct App {
    file: PathBuf,
    rx: mpsc::Receiver<()>,
    state: Option<State>,
    error: Option<anyhow::Error>,
}

pub fn run(file: PathBuf) -> Result<()> {
    // Watch the parent directory: many editors save by rename, which kills a
    // watch on the file itself.
    let abs = std::fs::canonicalize(&file)
        .with_context(|| format!("cannot open {}", file.display()))?;
    let dir = abs.parent().unwrap().to_path_buf();
    let (tx, rx) = mpsc::channel();
    let target = abs.clone();
    let mut watcher = notify::recommended_watcher(move |res: notify::Result<notify::Event>| {
        if let Ok(ev) = res {
            // Ignore Access events: our own read of the file would otherwise retrigger a reload forever.
            if !ev.kind.is_access() && ev.paths.iter().any(|p| p.file_name() == target.file_name()) {
                let _ = tx.send(());
            }
        }
    })?;
    watcher.watch(&dir, RecursiveMode::NonRecursive)?;

    let event_loop = EventLoop::new()?;
    event_loop.set_control_flow(ControlFlow::Poll);
    let mut app = App { file: abs, rx, state: None, error: None };
    event_loop.run_app(&mut app)?;
    drop(watcher);
    match app.error {
        Some(e) => Err(e),
        None => Ok(()),
    }
}

impl State {
    fn new(el: &ActiveEventLoop, file: &PathBuf) -> Result<Self> {
        let window = Arc::new(
            el.create_window(
                Window::default_attributes()
                    .with_title("shadertoy")
                    .with_inner_size(LogicalSize::new(1280.0, 720.0)),
            )?,
        );
        let instance = wgpu::Instance::default();
        let surface = instance.create_surface(window.clone())?;
        let mut gpu = Gpu::with_surface(&instance, Some(&surface))?;

        let size = window.inner_size();
        let mut config = surface
            .get_default_config(&gpu.adapter, size.width.max(1), size.height.max(1))
            .ok_or_else(|| anyhow!("surface is not supported by the GPU adapter"))?;
        // Shaders output display-ready values, so avoid sRGB surface formats
        // (they would gamma-encode a second time).
        let caps = surface.get_capabilities(&gpu.adapter);
        if let Some(f) = caps.formats.iter().copied().find(|f| !f.is_srgb()) {
            config.format = f;
        }
        surface.configure(&gpu.device, &config);
        gpu.set_format(config.format);

        let mut st = State {
            window,
            surface,
            config,
            gpu,
            status: String::new(),
            status_is_err: false,
            start: Instant::now(),
            paused_at: None,
            time_offset: Duration::ZERO,
            last_frame: Instant::now(),
            frame_no: 0,
            mouse: [0.0; 4],
            cursor: [0.0; 2],
            fps_ema: 60.0,
            render_ms_ema: 0.0,
            last_title: Instant::now(),
            last_overlay: None,
            last_err: None,
        };
        st.load(file);
        Ok(st)
    }

    fn load(&mut self, file: &PathBuf) {
        let res = std::fs::read_to_string(file)
            .map_err(|e| format!("read error: {e}"))
            .and_then(|src| self.gpu.set_shader(&src));
        match res {
            Ok(()) => {
                self.last_err = None;
                self.last_overlay = None; // bring the ms counter back right away
                self.status = "ok".into();
                self.status_is_err = false;
            }
            Err(e) => {
                if self.last_err.as_deref() != Some(e.as_str()) {
                    eprintln!("{}: {e}", file.display());
                    self.last_err = Some(e.clone());
                }
                self.status = e.lines().next().unwrap_or("error").to_string();
                self.status_is_err = true;
                self.show_error();
            }
        }
    }

    /// Pixels per font pixel for on-image text, scaled with the window height.
    fn text_scale(&self) -> u32 {
        (self.config.height / 440).max(1)
    }

    /// Draw the full compile error over the (last good) frame.
    fn show_error(&mut self) {
        let Some(err) = self.last_err.clone() else { return };
        let scale = self.text_scale();
        let cols = (self.config.width / (8 * scale)).saturating_sub(1) as usize;
        self.gpu.set_overlay(
            &format!("SHADER ERROR\n{err}"),
            [255, 255, 255],
            [150, 0, 0],
            scale,
            cols,
        );
    }

    fn resize(&mut self) {
        let size = self.window.inner_size();
        if size.width == 0 || size.height == 0 {
            return;
        }
        self.config.width = size.width;
        self.config.height = size.height;
        self.surface.configure(&self.gpu.device, &self.config);
        if self.status_is_err {
            self.show_error(); // re-wrap for the new window size
        }
    }

    fn toggle_pause(&mut self) {
        match self.paused_at.take() {
            Some(t) => self.time_offset += t.elapsed(),
            None => self.paused_at = Some(Instant::now()),
        }
    }

    fn restart(&mut self) {
        self.time_offset = self.start.elapsed();
        if let Some(_) = self.paused_at {
            let now = Instant::now();
            self.paused_at = Some(now);
            self.time_offset = now - self.start;
        }
        self.frame_no = 0;
    }

    fn redraw(&mut self) {
        let (w, h) = (self.config.width, self.config.height);
        let frame = match self.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(t) => t,
            wgpu::CurrentSurfaceTexture::Suboptimal(t) => {
                // Present this one, then reconfigure for the next.
                self.render_into(t, w, h);
                self.resize();
                return;
            }
            wgpu::CurrentSurfaceTexture::Outdated | wgpu::CurrentSurfaceTexture::Lost => {
                self.resize();
                return;
            }
            _ => return, // timeout / occluded / validation: try next frame
        };
        self.render_into(frame, w, h);
    }

    fn render_into(&mut self, frame: wgpu::SurfaceTexture, w: u32, h: u32) {
        let now = Instant::now();
        let delta = now.duration_since(self.last_frame).as_secs_f32();
        self.last_frame = now;
        if delta > 0.0 {
            self.fps_ema = self.fps_ema * 0.95 + (1.0 / delta) * 0.05;
        }
        let t = match self.paused_at {
            Some(p) => p - self.start - self.time_offset,
            None => self.start.elapsed() - self.time_offset,
        }
        .as_secs_f32();

        let view = frame.texture.create_view(&Default::default());
        // Submit, then wait for the GPU so we can time the actual render work
        // (without the wait we would only measure command submission).
        let render_start = Instant::now();
        self.gpu
            .render_to_view(&view, w, h, t, delta, self.frame_no, self.mouse);
        let _ = self.gpu.device.poll(wgpu::PollType::wait_indefinitely());
        let render_ms = render_start.elapsed().as_secs_f32() * 1000.0;
        self.render_ms_ema = if self.render_ms_ema == 0.0 {
            render_ms
        } else {
            self.render_ms_ema * 0.95 + render_ms * 0.05
        };
        if self.paused_at.is_none() {
            self.frame_no = self.frame_no.wrapping_add(1);
        }
        self.window.pre_present_notify();
        self.gpu.queue.present(frame);

        // On-image render time: whole milliseconds, refreshed every 5 s.
        if !self.status_is_err
            && self.last_overlay.map_or(true, |t| t.elapsed() > Duration::from_secs(5))
        {
            self.last_overlay = Some(Instant::now());
            let (scale, cols) = (self.text_scale(), 32);
            self.gpu.set_overlay(&format!("{:.0}ms", self.render_ms_ema), [150; 3], [0; 3], scale, cols);
        }

        if self.last_title.elapsed() > Duration::from_millis(500) {
            self.last_title = Instant::now();
            self.window.set_title(&format!(
                "shadertoy | {w}x{h} | render {:.2} ms | {:.0} fps | {}{}",
                self.render_ms_ema,
                self.fps_ema,
                self.status,
                if self.paused_at.is_some() { " | paused" } else { "" }
            ));
        }
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, el: &ActiveEventLoop) {
        if self.state.is_some() {
            return;
        }
        match State::new(el, &self.file) {
            Ok(s) => {
                s.window.request_redraw();
                self.state = Some(s);
            }
            Err(e) => {
                self.error = Some(e);
                el.exit();
            }
        }
    }

    fn window_event(&mut self, el: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        let Some(st) = self.state.as_mut() else { return };
        match event {
            WindowEvent::CloseRequested => el.exit(),
            WindowEvent::Resized(_) | WindowEvent::ScaleFactorChanged { .. } => st.resize(),
            WindowEvent::KeyboardInput { event, .. } if event.state == ElementState::Pressed => {
                match event.logical_key {
                    Key::Named(NamedKey::Escape) => el.exit(),
                    Key::Named(NamedKey::Space) => st.toggle_pause(),
                    Key::Character(ref c) if c.as_str() == "q" => el.exit(),
                    Key::Character(ref c) if c.as_str() == "r" => st.restart(),
                    _ => {}
                }
            }
            WindowEvent::CursorMoved { position, .. } => {
                // Shadertoy convention: origin at the bottom-left.
                let x = position.x as f32;
                let y = st.config.height as f32 - position.y as f32;
                st.cursor = [x, y];
                if st.mouse[2] > 0.0 {
                    st.mouse[0] = x;
                    st.mouse[1] = y;
                }
            }
            WindowEvent::MouseInput { state, button: MouseButton::Left, .. } => {
                let [x, y] = st.cursor;
                match state {
                    ElementState::Pressed => st.mouse = [x, y, x, y],
                    // Shadertoy: sign of zw encodes button state.
                    ElementState::Released => {
                        st.mouse[2] = -st.mouse[2].abs();
                        st.mouse[3] = -st.mouse[3].abs();
                    }
                }
            }
            WindowEvent::RedrawRequested => st.redraw(),
            _ => {}
        }
    }

    fn about_to_wait(&mut self, _el: &ActiveEventLoop) {
        let Some(st) = self.state.as_mut() else { return };
        // Hot reload (debounced: drain everything queued, then reload once).
        if self.rx.try_recv().is_ok() {
            while self.rx.try_recv().is_ok() {}
            std::thread::sleep(Duration::from_millis(20)); // let the write finish
            while self.rx.try_recv().is_ok() {}
            st.load(&self.file);
        }
        st.window.request_redraw(); // vsync (Fifo present mode) paces the loop
    }
}
