use anyhow::{anyhow, Result};
use std::borrow::Cow;
use std::path::{Path, PathBuf};
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
// iChannel0: in the Image pass, the latest output of Buffer A; in Buffer A,
// its own previous frame. Use texelFetch(iChannel0, ivec2(x, y), 0).
layout(set = 1, binding = 0) uniform texture2D _ch0_tex;
layout(set = 1, binding = 1) uniform sampler _ch0_smp;
#define iChannel0 sampler2D(_ch0_tex, _ch0_smp)
layout(location = 0) out vec4 _fragColor;
";
const PROLOGUE_LINES: u32 = 24;

const EPILOGUE: &str = "
void main() {
    vec4 c = vec4(0.0, 0.0, 0.0, 1.0);
    mainImage(c, vec2(gl_FragCoord.x, u_res.y - gl_FragCoord.y));
    _fragColor = vec4(c.rgb, 1.0);
}
";

/// Buffer passes write raw float data: no alpha forcing, and fragCoord.y is the
/// texel row (not flipped), so `texelFetch(iChannel0, ivec2(fragCoord), 0)` in
/// the pass reads back exactly what that pixel wrote last frame.
const BUFFER_EPILOGUE: &str = "
void main() {
    vec4 c = vec4(0.0);
    mainImage(c, gl_FragCoord.xy);
    _fragColor = c;
}
";

/// Buffers hold data (positions etc.), so they need float precision.
const BUFFER_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba32Float;

const VERTEX: &str = "
@vertex
fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
    var p = array<vec2<f32>, 3>(vec2(-1.0, -1.0), vec2(3.0, -1.0), vec2(-1.0, 3.0));
    return vec4(p[i], 0.0, 1.0);
}
";

/// Draws the overlay bitmap (stats / error text) over the shader output.
const OVERLAY_WGSL: &str = "
struct P { scale: u32, w: u32, h: u32, pad: u32 };
@group(0) @binding(0) var<uniform> prm: P;
@group(0) @binding(1) var tex: texture_2d<f32>;
@vertex
fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
    var p = array<vec2<f32>, 3>(vec2(-1.0, -1.0), vec2(3.0, -1.0), vec2(-1.0, 3.0));
    return vec4(p[i], 0.0, 1.0);
}
@fragment
fn fs(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
    let q = vec2<u32>(pos.xy) / prm.scale;
    if (q.x >= prm.w || q.y >= prm.h) { discard; }
    return textureLoad(tex, vec2<i32>(q), 0);
}
";

struct Overlay {
    bind_group: wgpu::BindGroup,
}

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

/// Buffer A: two canvas-sized float textures that swap roles every frame
/// (a pass cannot read and write the same texture).
struct Buffer {
    w: u32,
    h: u32,
    views: [wgpu::TextureView; 2],
    /// binds[i] exposes texture i as iChannel0.
    binds: [wgpu::BindGroup; 2],
    /// Index of the texture holding the most recently written frame.
    cur: usize,
    /// Frames rendered since the buffer was (re)created; used as iFrame in the
    /// buffer pass so `iFrame == 0` means "initialise" even after reload/resize.
    frames: i32,
}

pub struct Gpu {
    pub adapter: wgpu::Adapter,
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub adapter_name: String,
    vs: wgpu::ShaderModule,
    bind_group: wgpu::BindGroup,
    ubuf: wgpu::Buffer,
    pipeline: Option<wgpu::RenderPipeline>,
    /// Colour format the fragment pipeline renders to.
    format: wgpu::TextureFormat,
    target: Option<Target>,
    pixels: Vec<u8>,
    overlay_bgl: wgpu::BindGroupLayout,
    overlay_pipeline: Option<wgpu::RenderPipeline>,
    overlay: Option<Overlay>,
    pipeline_layout: wgpu::PipelineLayout,
    chan_bgl: wgpu::BindGroupLayout,
    sampler: wgpu::Sampler,
    /// 1x1 stand-in for iChannel0 when there is no buffer.
    dummy_bind: wgpu::BindGroup,
    ubuf_a: wgpu::Buffer,
    bind_group_a: wgpu::BindGroup,
    buffer_pipeline: Option<wgpu::RenderPipeline>,
    buffer: Option<Buffer>,
}

/// `foo.glsl` -> `foo.<kind>.glsl` (kind: "bufferA" or "common")
fn sibling_path(main: &Path, kind: &str) -> PathBuf {
    let stem = main.file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default();
    main.with_file_name(format!("{stem}.{kind}.glsl"))
}

/// File names used to label shader errors.
pub struct Names {
    pub image: String,
    pub buffer: String,
    pub common: String,
}

fn file_name(p: &Path) -> String {
    p.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default()
}

/// `foo.glsl` -> `foo.bufferA.glsl`
pub fn buffer_path(main: &Path) -> PathBuf {
    sibling_path(main, "bufferA")
}

/// `foo.glsl` -> `foo.common.glsl` (prepended to every pass, like Shadertoy's Common tab)
pub fn common_path(main: &Path) -> PathBuf {
    sibling_path(main, "common")
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

        let overlay_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
            ],
        });

        let chan_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::NonFiltering),
                    count: None,
                },
            ],
        });
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor::default());
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[Some(&bgl), Some(&chan_bgl)],
            immediate_size: 0,
        });
        let dummy_tex = device.create_texture(&wgpu::TextureDescriptor {
            label: None,
            size: wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: BUFFER_FORMAT,
            usage: wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let dummy_bind = chan_bind(&device, &chan_bgl, &sampler, &dummy_tex.create_view(&Default::default()));
        let ubuf_a = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: std::mem::size_of::<Uniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let bind_group_a = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &bgl,
            entries: &[wgpu::BindGroupEntry { binding: 0, resource: ubuf_a.as_entire_binding() }],
        });

        Ok(Self {
            adapter,
            device,
            queue,
            adapter_name,
            vs,
            bind_group,
            ubuf,
            pipeline: None,
            format: FORMAT,
            target: None,
            pixels: Vec::new(),
            overlay_bgl,
            overlay_pipeline: None,
            overlay: None,
            pipeline_layout,
            chan_bgl,
            sampler,
            dummy_bind,
            ubuf_a,
            bind_group_a,
            buffer_pipeline: None,
            buffer: None,
        })
    }

    /// Load `main` plus its optional siblings `<name>.bufferA.glsl` and
    /// `<name>.common.glsl` (code shared by every pass). Both are
    /// compiled before anything is swapped in, so an error in either keeps the
    /// last good frame rendering.
    pub fn load_files(&mut self, main: &Path) -> std::result::Result<(), String> {
        let image = std::fs::read_to_string(main).map_err(|e| format!("{}: read error: {e}", file_name(main)))?;
        let optional = |path: PathBuf| match std::fs::read_to_string(&path) {
            Ok(src) => Ok(Some(src)),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
            Err(e) => Err(format!("{}: read error: {e}", file_name(&path))),
        };
        let buffer = optional(buffer_path(main))?;
        let common = optional(common_path(main))?.unwrap_or_default();
        let names = Names {
            image: file_name(main),
            buffer: file_name(&buffer_path(main)),
            common: file_name(&common_path(main)),
        };
        self.set_shaders(&image, buffer.as_deref(), &common, &names)
    }

    /// Compile the Image shader and an optional Buffer A shader.
    pub fn set_shaders(
        &mut self,
        image_src: &str,
        buffer_src: Option<&str>,
        common: &str,
        names: &Names,
    ) -> std::result::Result<(), String> {
        let image_fs = self.compile(image_src, common, EPILOGUE, &names.image, &names.common)?;
        let buffer_fs = match buffer_src {
            Some(src) => Some(self.compile(src, common, BUFFER_EPILOGUE, &names.buffer, &names.common)?),
            None => None,
        };
        self.pipeline = Some(self.make_pipeline(&image_fs, self.format));
        self.buffer_pipeline = buffer_fs.map(|fs| self.make_pipeline(&fs, BUFFER_FORMAT));
        self.buffer = None; // start from zeroed textures with iFrame == 0
        Ok(())
    }

    /// Compile a Shadertoy-style fragment shader. Errors carry user-file line numbers.
    /// Errors are reported as `file:line: message`, where `file` is `src_name`
    /// for the pass source and `common_name` for the shared common code.
    fn compile(
        &self,
        user_src: &str,
        common: &str,
        epilogue: &str,
        src_name: &str,
        common_name: &str,
    ) -> std::result::Result<wgpu::ShaderModule, String> {
        if !user_src.contains("mainImage") {
            return Err(format!("{src_name}: no mainImage(out vec4, in vec2) function found"));
        }
        // Common code goes between the prologue and the pass source.
        let common = common.trim_end_matches('\n');
        let common_lines = common.lines().count() as u32;
        let common_block = if common.is_empty() { String::new() } else { format!("{common}\n") };
        let full = format!("{PROLOGUE}{common_block}{user_src}\n{epilogue}");
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
                        if line <= common_lines {
                            format!("{common_name}:{line}: {}", e.kind)
                        } else {
                            format!("{src_name}:{}: {}", line - common_lines, e.kind)
                        }
                    })
                    .collect::<Vec<_>>()
                    .join("\n")
            })?;
        fix_unemitted_call_pointers(&mut module);
        naga::valid::Validator::new(naga::valid::ValidationFlags::all(), naga::valid::Capabilities::all())
            .validate(&module)
            .map_err(|e| format!("{src_name}: {}", e.as_inner()))?;

        Ok(self.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("fs"),
            source: wgpu::ShaderSource::Naga(Cow::Owned(module)),
        }))
    }

    fn make_pipeline(&self, fs: &wgpu::ShaderModule, format: wgpu::TextureFormat) -> wgpu::RenderPipeline {
        self.device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: None,
            layout: Some(&self.pipeline_layout),
            vertex: wgpu::VertexState {
                module: &self.vs,
                entry_point: Some("vs"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            fragment: Some(wgpu::FragmentState {
                module: fs,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                targets: &[Some(format.into())],
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        })
    }

    fn make_buffer(&self, w: u32, h: u32) -> Buffer {
        let mk = || {
            self.device.create_texture(&wgpu::TextureDescriptor {
                label: Some("buffer A"),
                size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: BUFFER_FORMAT,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            })
        };
        let views = [mk().create_view(&Default::default()), mk().create_view(&Default::default())];
        let binds = [
            chan_bind(&self.device, &self.chan_bgl, &self.sampler, &views[0]),
            chan_bind(&self.device, &self.chan_bgl, &self.sampler, &views[1]),
        ];
        Buffer { w, h, views, binds, cur: 0, frames: 0 }
    }

    /// Run the Buffer A pass (if there is one): read texture `cur`, write the
    /// other one, then swap. Must be encoded before the Image pass.
    fn run_buffer(&mut self, enc: &mut wgpu::CommandEncoder, w: u32, h: u32, time: f32, delta: f32, mouse: [f32; 4]) {
        let Some(pipe) = self.buffer_pipeline.clone() else {
            self.buffer = None;
            return;
        };
        if !matches!(&self.buffer, Some(b) if b.w == w && b.h == h) {
            self.buffer = Some(self.make_buffer(w, h));
        }
        let b = self.buffer.as_mut().unwrap();
        let u = Self::uniforms(w, h, time, delta, b.frames, mouse);
        self.queue.write_buffer(&self.ubuf_a, 0, bytemuck::bytes_of(&u));
        let dst = 1 - b.cur;
        {
            let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("buffer A"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &b.views[dst],
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
            pass.set_pipeline(&pipe);
            pass.set_bind_group(0, &self.bind_group_a, &[]);
            pass.set_bind_group(1, &b.binds[b.cur], &[]);
            pass.draw(0..3, 0..1);
        }
        b.cur = dst;
        b.frames = b.frames.wrapping_add(1);
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

    /// Draw `text` (ASCII, `\n` separated, 8x8 font) in the top-left corner
    /// over the shader output, `scale` pixels per font pixel. Lines longer than
    /// `max_cols` wrap. An empty string removes the overlay.
    pub fn set_overlay(&mut self, text: &str, fg: [u8; 3], bg: [u8; 3], scale: u32, max_cols: usize) {
        const MAX_LINES: usize = 60;
        if text.is_empty() {
            self.overlay = None;
            return;
        }
        let max_cols = max_cols.max(4);
        // Wrap long lines instead of cutting them off.
        let mut lines: Vec<String> = Vec::new();
        for l in text.lines() {
            let l: Vec<u8> = l
                .chars()
                .map(|c| if c == '\t' { ' ' } else if c.is_ascii() && !c.is_control() { c } else { '?' } as u8)
                .collect();
            if l.is_empty() {
                lines.push(String::new());
            }
            for chunk in l.chunks(max_cols) {
                lines.push(String::from_utf8_lossy(chunk).into_owned());
            }
        }
        lines.truncate(MAX_LINES);
        if lines.is_empty() {
            lines.push(String::new());
        }
        let cols = lines.iter().map(|l| l.len()).max().unwrap_or(1).max(1) as u32;
        let pad = 4u32;
        let (w, h) = (cols * 8 + 2 * pad, lines.len() as u32 * 9 + 2 * pad);
        let mut data = vec![0u8; (w * h * 4) as usize];
        for px in data.chunks_exact_mut(4) {
            px.copy_from_slice(&[bg[0], bg[1], bg[2], 215]);
        }
        for (row, line) in lines.iter().enumerate() {
            for (col, ch) in line.bytes().enumerate() {
                let glyph = font8x8::legacy::BASIC_LEGACY[(ch & 0x7f) as usize];
                for (gy, bits) in glyph.iter().enumerate() {
                    for gx in 0..8u32 {
                        if bits >> gx & 1 == 1 {
                            let x = pad + col as u32 * 8 + gx;
                            let y = pad + row as u32 * 9 + gy as u32;
                            let i = ((y * w + x) * 4) as usize;
                            data[i..i + 4].copy_from_slice(&[fg[0], fg[1], fg[2], 255]);
                        }
                    }
                }
            }
        }

        let tex = self.device.create_texture(&wgpu::TextureDescriptor {
            label: None,
            size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        self.queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &tex,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &data,
            wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(w * 4), rows_per_image: Some(h) },
            wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
        );
        let ubuf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: 16,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.queue.write_buffer(&ubuf, 0, bytemuck::bytes_of(&[scale.max(1), w, h, 0u32]));
        let view = tex.create_view(&Default::default());
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.overlay_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: ubuf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&view) },
            ],
        });
        self.overlay = Some(Overlay { bind_group });

        if self.overlay_pipeline.is_none() {
            let module = self.device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("overlay"),
                source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(OVERLAY_WGSL)),
            });
            let layout = self.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: None,
                bind_group_layouts: &[Some(&self.overlay_bgl)],
                immediate_size: 0,
            });
            self.overlay_pipeline = Some(self.device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: None,
                layout: Some(&layout),
                vertex: wgpu::VertexState {
                    module: &module,
                    entry_point: Some("vs"),
                    compilation_options: Default::default(),
                    buffers: &[],
                },
                fragment: Some(wgpu::FragmentState {
                    module: &module,
                    entry_point: Some("fs"),
                    compilation_options: Default::default(),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: self.format,
                        blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                }),
                primitive: wgpu::PrimitiveState::default(),
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
                multiview_mask: None,
                cache: None,
            }));
        }
    }

    /// Set the colour format of the render pipeline (window mode: the surface
    /// format). Takes effect for shaders compiled afterwards.
    pub fn set_format(&mut self, format: wgpu::TextureFormat) {
        self.format = format;
        self.overlay_pipeline = None;
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
        self.run_buffer(&mut enc, w, h, time, delta, mouse);
        self.draw_pass(&mut enc, view);
        self.queue.submit([enc.finish()]);
    }

    fn write_uniforms(&self, w: u32, h: u32, time: f32, delta: f32, frame: i32, mouse: [f32; 4]) {
        let u = Self::uniforms(w, h, time, delta, frame, mouse);
        self.queue.write_buffer(&self.ubuf, 0, bytemuck::bytes_of(&u));
    }

    fn uniforms(w: u32, h: u32, time: f32, delta: f32, frame: i32, mouse: [f32; 4]) -> Uniforms {
        let date = {
            let secs = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_secs_f32())
                .unwrap_or(0.0);
            [0.0, 0.0, 0.0, secs % 86400.0]
        };
        Uniforms {
            res: [w as f32, h as f32, 1.0, 0.0],
            mouse,
            date,
            time,
            delta,
            frame,
            _pad: 0.0,
        }
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
                let chan = self.buffer.as_ref().map_or(&self.dummy_bind, |b| &b.binds[b.cur]);
                pass.set_bind_group(1, chan, &[]);
                pass.draw(0..3, 0..1);
            }
            if let (Some(p), Some(o)) = (&self.overlay_pipeline, &self.overlay) {
                pass.set_pipeline(p);
                pass.set_bind_group(0, &o.bind_group, &[]);
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
        self.run_buffer(&mut enc, w, h, time, delta, mouse);
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

fn chan_bind(
    device: &wgpu::Device,
    bgl: &wgpu::BindGroupLayout,
    sampler: &wgpu::Sampler,
    view: &wgpu::TextureView,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: bgl,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(view) },
            wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(sampler) },
        ],
    })
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
