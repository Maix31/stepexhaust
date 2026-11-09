use anyhow::{anyhow, Result};
use std::borrow::Cow;
use wgpu::naga;

/// Number of lines in PROLOGUE, so error line numbers match the user's file.
const PROLOGUE: &str = "#version 450
precision highp float;
precision highp int;
layout(set = 0, binding = 0) uniform ShadertoyUniforms {
    vec4 u_res;
    vec4 u_mouse;
    vec4 u_date;
    float u_time;
    float u_delta;
    int u_frame;
    float u_pad;
};
#define iResolution u_res.xyz
#define iMouse u_mouse
#define iDate u_date
#define iTime u_time
#define iTimeDelta u_delta
#define iFrame u_frame
layout(location = 0) out vec4 _fragColor;
";
const PROLOGUE_LINES: u32 = 19;

const EPILOGUE: &str = "
void main() {
    vec4 c = vec4(0.0, 0.0, 0.0, 1.0);
    mainImage(c, vec2(gl_FragCoord.x, u_res.y - gl_FragCoord.y));
    _fragColor = vec4(c.rgb, 1.0);
}
";

const VERTEX: &str = "
@vertex
fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
    var p = array<vec2<f32>, 3>(vec2(-1.0, -1.0), vec2(3.0, -1.0), vec2(-1.0, 3.0));
    return vec4(p[i], 0.0, 1.0);
}
";

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Uniforms {
    res: [f32; 4],
    mouse: [f32; 4],
    date: [f32; 4],
    time: f32,
    delta: f32,
    frame: i32,
    _pad: f32,
}

struct Target {
    w: u32,
    h: u32,
    tex: wgpu::Texture,
    view: wgpu::TextureView,
    readback: wgpu::Buffer,
    padded_row: u32,
}

pub struct Gpu {
    pub adapter: wgpu::Adapter,
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub adapter_name: String,
    vs: wgpu::ShaderModule,
    bgl: wgpu::BindGroupLayout,
    bind_group: wgpu::BindGroup,
    ubuf: wgpu::Buffer,
    pipeline: Option<wgpu::RenderPipeline>,
    /// Colour format the fragment pipeline renders to.
    format: wgpu::TextureFormat,
    target: Option<Target>,
    pixels: Vec<u8>,
}

const FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;

impl Gpu {
    /// Headless GPU (renders to an offscreen texture, read back to the CPU).
    pub fn new() -> Result<Self> {
        Self::with_surface(&wgpu::Instance::default(), None)
    }

    /// GPU compatible with `surface` (for window mode).
    pub fn with_surface(instance: &wgpu::Instance, surface: Option<&wgpu::Surface>) -> Result<Self> {
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            force_fallback_adapter: false,
            compatible_surface: surface,
            ..Default::default()
        }))
        .or_else(|_| {
            pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
                force_fallback_adapter: true,
                ..Default::default()
            }))
        })
        .map_err(|e| anyhow!("no GPU adapter found (Vulkan/GL driver missing?): {e}"))?;
        let adapter_name = adapter.get_info().name;

        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default()))?;

        let vs = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("vs"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(VERTEX)),
        });
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });
        let ubuf = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: std::mem::size_of::<Uniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &bgl,
            entries: &[wgpu::BindGroupEntry { binding: 0, resource: ubuf.as_entire_binding() }],
        });

        Ok(Self {
            adapter,
            device,
            queue,
            adapter_name,
            vs,
            bgl,
            bind_group,
            ubuf,
            pipeline: None,
            format: FORMAT,
            target: None,
            pixels: Vec::new(),
        })
    }

    /// Compile a Shadertoy-style fragment shader. On failure the previous
    /// pipeline is kept, so the last good frame keeps rendering.
    pub fn set_shader(&mut self, user_src: &str) -> std::result::Result<(), String> {
        if !user_src.contains("mainImage") {
            return Err("no mainImage(out vec4, in vec2) function found".into());
        }
        let full = format!("{PROLOGUE}{user_src}\n{EPILOGUE}");
        debug_assert_eq!(PROLOGUE.lines().count() as u32, PROLOGUE_LINES);

        let mut frontend = naga::front::glsl::Frontend::default();
        let mut module = frontend
            .parse(&naga::front::glsl::Options::from(naga::ShaderStage::Fragment), &full)
            .map_err(|errs| {
                errs.errors
                    .iter()
                    .map(|e| {
                        let line = e
                            .meta
                            .location(&full)
                            .line_number
                            .saturating_sub(PROLOGUE_LINES);
                        format!("line {line}: {}", e.kind)
                    })
                    .collect::<Vec<_>>()
                    .join("\n")
            })?;
        fix_unemitted_call_pointers(&mut module);
        naga::valid::Validator::new(naga::valid::ValidationFlags::all(), naga::valid::Capabilities::all())
            .validate(&module)
            .map_err(|e| format!("{}", e.as_inner()))?;

        let fs = self.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("fs"),
            source: wgpu::ShaderSource::Naga(Cow::Owned(module)),
        });
        let layout = self.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[Some(&self.bgl)],
            immediate_size: 0,
        });
        let pipeline = self.device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: None,
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module: &self.vs,
                entry_point: Some("vs"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            fragment: Some(wgpu::FragmentState {
                module: &fs,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                targets: &[Some(self.format.into())],
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });
        self.pipeline = Some(pipeline);
        Ok(())
    }

    fn ensure_target(&mut self, w: u32, h: u32) {
        if matches!(&self.target, Some(t) if t.w == w && t.h == h) {
            return;
        }
        let tex = self.device.create_texture(&wgpu::TextureDescriptor {
            label: None,
            size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let view = tex.create_view(&Default::default());
        let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
        let padded_row = (w * 4).div_ceil(align) * align;
        let readback = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: (padded_row * h) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        self.target = Some(Target { w, h, tex, view, readback, padded_row });
    }

    /// Set the colour format of the render pipeline (window mode: the surface
    /// format). Takes effect for shaders compiled afterwards.
    pub fn set_format(&mut self, format: wgpu::TextureFormat) {
        self.format = format;
    }

    /// Render one frame straight into `view` (e.g. a swapchain texture).
    pub fn render_to_view(
        &mut self,
        view: &wgpu::TextureView,
        w: u32,
        h: u32,
        time: f32,
        delta: f32,
        frame: i32,
        mouse: [f32; 4],
    ) {
        self.write_uniforms(w, h, time, delta, frame, mouse);
        let mut enc = self.device.create_command_encoder(&Default::default());
        self.draw_pass(&mut enc, view);
        self.queue.submit([enc.finish()]);
    }

    fn write_uniforms(&self, w: u32, h: u32, time: f32, delta: f32, frame: i32, mouse: [f32; 4]) {
        let date = {
            let secs = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_secs_f32())
                .unwrap_or(0.0);
            [0.0, 0.0, 0.0, secs % 86400.0]
        };
        let u = Uniforms {
            res: [w as f32, h as f32, 1.0, 0.0],
            mouse,
            date,
            time,
            delta,
            frame,
            _pad: 0.0,
        };
        self.queue.write_buffer(&self.ubuf, 0, bytemuck::bytes_of(&u));
    }

    fn draw_pass(&self, enc: &mut wgpu::CommandEncoder, view: &wgpu::TextureView) {
        {
            let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: None,
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            if let Some(p) = &self.pipeline {
                pass.set_pipeline(p);
                pass.set_bind_group(0, &self.bind_group, &[]);
                pass.draw(0..3, 0..1);
            }
        }
    }

    /// Render one frame and return tightly packed RGBA8 pixels (top row first).
    pub fn render(
        &mut self,
        w: u32,
        h: u32,
        time: f32,
        delta: f32,
        frame: i32,
        mouse: [f32; 4],
    ) -> Result<&[u8]> {
        let (w, h) = (w.max(1), h.max(1));
        self.ensure_target(w, h);
        self.write_uniforms(w, h, time, delta, frame, mouse);
        let mut enc = self.device.create_command_encoder(&Default::default());
        self.draw_pass(&mut enc, &self.target.as_ref().unwrap().view);
        let t = self.target.as_ref().unwrap();
        enc.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &t.tex,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &t.readback,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(t.padded_row),
                    rows_per_image: Some(h),
                },
            },
            wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
        );
        self.queue.submit([enc.finish()]);

        let slice = t.readback.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        self.device.poll(wgpu::PollType::wait_indefinitely())?;
        rx.recv()??;
        {
            let data = slice.get_mapped_range()?;
            self.pixels.clear();
            for row in 0..h as usize {
                let s = row * t.padded_row as usize;
                self.pixels.extend_from_slice(&data[s..s + (w * 4) as usize]);
            }
        }
        t.readback.unmap();
        Ok(&self.pixels)
    }
}

/// Work around a naga bug: the SPIR-V backend panics ("Expression [n] is not
/// cached!") when a call passes a pointer into part of a variable, e.g.
/// `f(r.m)` with an `out` parameter, because it only expects whole-variable
/// pointers there. Rewrite such calls to use a temporary local:
///   tmp = r.m; f(tmp); r.m = tmp;
fn fix_unemitted_call_pointers(module: &mut naga::Module) {
    use naga::{Block, Expression, Handle, Range, Span, Statement};

    // Pointee type of each pointer-typed parameter, per function.
    let pointees: Vec<(Handle<naga::Function>, Vec<Option<Handle<naga::Type>>>)> = module
        .functions
        .iter()
        .map(|(h, f)| {
            let v = f
                .arguments
                .iter()
                .map(|a| match module.types[a.ty].inner {
                    naga::TypeInner::Pointer { base, .. } => Some(base),
                    _ => None,
                })
                .collect();
            (h, v)
        })
        .collect();

    fn fix(
        block: &mut Block,
        f: &mut (naga::Arena<Expression>, naga::Arena<naga::LocalVariable>),
        pointees: &[(Handle<naga::Function>, Vec<Option<Handle<naga::Type>>>)],
    ) {
        let old = std::mem::take(block);
        for (mut stmt, span) in old.span_into_iter() {
            match &mut stmt {
                Statement::Call { function, arguments, .. } => {
                    let params = &pointees.iter().find(|(h, _)| h == function).unwrap().1;
                    let mut post = Vec::new();
                    for (i, arg) in arguments.iter_mut().enumerate() {
                        let Some(base_ty) = params.get(i).copied().flatten() else { continue };
                        if !matches!(
                            f.0[*arg],
                            Expression::Access { .. } | Expression::AccessIndex { .. }
                        ) {
                            continue;
                        }
                        let member = *arg;
                        let var = f.1.append(
                            naga::LocalVariable { name: None, ty: base_ty, init: None },
                            Span::UNDEFINED,
                        );
                        let tmp = f.0.append(Expression::LocalVariable(var), Span::UNDEFINED);
                        // Copy in (covers inout), then copy out after the call.
                        let l_in = f.0.append(Expression::Load { pointer: member }, Span::UNDEFINED);
                        block.push(Statement::Emit(Range::new_from_bounds(l_in, l_in)), span);
                        block.push(Statement::Store { pointer: tmp, value: l_in }, span);
                        let l_out = f.0.append(Expression::Load { pointer: tmp }, Span::UNDEFINED);
                        post.push(Statement::Emit(Range::new_from_bounds(l_out, l_out)));
                        post.push(Statement::Store { pointer: member, value: l_out });
                        *arg = tmp;
                    }
                    block.push(stmt, span);
                    for s in post {
                        block.push(s, span);
                    }
                    continue;
                }
                Statement::Block(b) => fix(b, f, pointees),
                Statement::If { accept, reject, .. } => {
                    fix(accept, f, pointees);
                    fix(reject, f, pointees);
                }
                Statement::Switch { cases, .. } => {
                    cases.iter_mut().for_each(|c| fix(&mut c.body, f, pointees))
                }
                Statement::Loop { body, continuing, .. } => {
                    fix(body, f, pointees);
                    fix(continuing, f, pointees);
                }
                _ => {}
            }
            block.push(stmt, span);
        }
    }

    let run = |func: &mut naga::Function| {
        let mut body = std::mem::take(&mut func.body);
        let mut arenas = (
            std::mem::take(&mut func.expressions),
            std::mem::take(&mut func.local_variables),
        );
        fix(&mut body, &mut arenas, &pointees);
        func.body = body;
        func.expressions = arenas.0;
        func.local_variables = arenas.1;
    };
    for (_, func) in module.functions.iter_mut() {
        run(func);
    }
    for ep in module.entry_points.iter_mut() {
        run(&mut ep.function);
    }
}
