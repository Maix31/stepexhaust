#define MAX_DISTANCE 1e2

#define NO_HIT_STEP_EXHAUST_MAT -2
#define NO_HIT_DISTANCE_EXHAUST_MAT -1

// Layout, DirectionalLight, FLOOR_* ... come from sphere.common.glsl

struct RayMarchResult
{
    vec3 p;
    float min_d;
    int i;
};

float sphere(vec3 p, float r)
{
    return length(p) - r;
}

// Nearest-surface distance via the BVH in Buffer A (see sphere.common.glsl).
// Exact: a subtree is skipped only if its bounding sphere is already farther
// than the best distance found, so it cannot contain anything nearer.
vec4 node_bound(int k)
{
    return texelFetch(iChannel0, texel_pos(NODE_BASE + k), 0);
}

float map(vec3 p, out int i_out)
{
    float best = length(p - FLOOR_SPHERE.xyz) - FLOOR_SPHERE.w;
    i_out = FLOOR_ID;

    int   stack_k[16];   // pending nodes ...
    float stack_d[16];   // ... and their lower-bound distance
    int sp = 1;
    stack_k[0] = 1;
    vec4 root = node_bound(1);
    stack_d[0] = length(p - root.xyz) - root.w;

    while (sp > 0)
    {
        sp--;
        int k = stack_k[sp];
        float d = stack_d[sp];
        if (d >= best) continue;            // cannot beat what we have

        if (k >= SPHERE_COUNT)              // leaf: its bound IS the sphere
        {
            best = d;
            i_out = k - SPHERE_COUNT;
            continue;
        }

        vec4 b0 = node_bound(2 * k);
        vec4 b1 = node_bound(2 * k + 1);
        float d0 = length(p - b0.xyz) - b0.w;
        float d1 = length(p - b1.xyz) - b1.w;
        // Push the farther child first so the nearer one is visited next.
        if (d0 < d1)
        {
            if (d1 < best) { stack_k[sp] = 2 * k + 1; stack_d[sp] = d1; sp++; }
            if (d0 < best) { stack_k[sp] = 2 * k;     stack_d[sp] = d0; sp++; }
        }
        else
        {
            if (d0 < best) { stack_k[sp] = 2 * k;     stack_d[sp] = d0; sp++; }
            if (d1 < best) { stack_k[sp] = 2 * k + 1; stack_d[sp] = d1; sp++; }
        }
    }
    return best;
}

int material_of(int i)
{
    if (i == FLOOR_ID) return FLOOR_MAT;
    return int(texelFetch(iChannel0, texel_pos(ATTR_BASE + i), 0).w);
}

vec3 normal(vec3 p)
{
    float e = 1e-3;
    int m;

    return normalize(vec3(
        map(p + vec3(e,0,0), m) -
        map(p - vec3(e,0,0), m),

        map(p + vec3(0,e,0), m) -
        map(p - vec3(0,e,0), m),

        map(p + vec3(0,0,e), m) -
        map(p - vec3(0,0,e), m)
    ));
}

bool is_no_hit(int m)
{
    return m == NO_HIT_DISTANCE_EXHAUST_MAT || m == NO_HIT_STEP_EXHAUST_MAT; 
}

RayMarchResult ray_march(vec3 ro, vec3 rd, int max_steps, float max_distance)
{
    vec3 c = vec3(0);
    RayMarchResult r;
    r.p = ro;
    r.min_d = 1e20;
    float t = 0.;
    for (int i = 0; i < max_steps; i++)
    {
        float d = map(r.p, r.i);
        float eps = max(1e-4, 0.5 * t / iResolution.y);

        r.p += rd * d;
        t += d;

        r.min_d = min(d / t, r.min_d);

        if (d < eps)
        {
            return r;
        }
        
        if (t > max_distance)
        {
            r.i = NO_HIT_DISTANCE_EXHAUST_MAT;
            return r;
        }
    }

    r.i = NO_HIT_STEP_EXHAUST_MAT;
    return r;
}

// Entry distance of the ray into a sphere (center.xyz, radius), clamped to >= 0
// when the origin is inside it; -1 if the ray misses it or it is fully behind.
float ray_sphere(vec3 ro, vec3 rd, vec4 s)
{
    vec3 oc = ro - s.xyz;
    float b = dot(oc, rd);
    float h = b * b - (dot(oc, oc) - s.w * s.w);
    if (h < 0.) return -1.;
    h = sqrt(h);
    if (-b + h < 0.) return -1.;
    return max(-b - h, 0.);
}

// Exact closest hit along the ray (rd normalized) within max_t, found by one
// BVH traversal: no stepping. Nearer child first; a subtree is skipped when the
// ray misses its bounding sphere or enters it beyond the best hit so far.
// r.min_d is not computed (0): use ray_march when you need the soft-shadow term.
RayMarchResult ray_cast(vec3 ro, vec3 rd, float max_t)
{
    RayMarchResult r;
    r.min_d = 0.;
    r.i = NO_HIT_DISTANCE_EXHAUST_MAT;
    float best = max_t;

    float tf = ray_sphere(ro, rd, FLOOR_SPHERE);
    if (tf >= 0. && tf < best) { best = tf; r.i = FLOOR_ID; }

    int   stack_k[16];
    float stack_t[16];
    int sp = 0;
    float t_root = ray_sphere(ro, rd, node_bound(1));
    if (t_root >= 0. && t_root < best) { stack_k[0] = 1; stack_t[0] = t_root; sp = 1; }

    while (sp > 0)
    {
        sp--;
        int k = stack_k[sp];
        float t = stack_t[sp];
        if (t >= best) continue;

        if (k >= SPHERE_COUNT)              // leaf: its bound IS the sphere
        {
            best = t;
            r.i = k - SPHERE_COUNT;
            continue;
        }

        float t0 = ray_sphere(ro, rd, node_bound(2 * k));
        float t1 = ray_sphere(ro, rd, node_bound(2 * k + 1));
        bool h0 = t0 >= 0. && t0 < best;
        bool h1 = t1 >= 0. && t1 < best;
        // Push the farther child first so the nearer one is visited next.
        if (t0 < t1)
        {
            if (h1) { stack_k[sp] = 2 * k + 1; stack_t[sp] = t1; sp++; }
            if (h0) { stack_k[sp] = 2 * k;     stack_t[sp] = t0; sp++; }
        }
        else
        {
            if (h0) { stack_k[sp] = 2 * k;     stack_t[sp] = t0; sp++; }
            if (h1) { stack_k[sp] = 2 * k + 1; stack_t[sp] = t1; sp++; }
        }
    }
    r.p = ro + rd * best;
    return r;
}

vec3 phong_bling(in DirectionalLight dl, vec3 p)
{
    float d = max(dot(normal(p), dl.d), 0.);
    vec3 c = vec3(0.3, 0.5, 0.7);
    return mix(c, dl.c ,d);  
}

vec3 checker(vec3 p)
{
    vec2 g = floor(p.xz);
    return vec3(mod(g.x + g.y, 2.0));
}

vec3 color(int i, in DirectionalLight dl, vec3 p)
{
    // Rays that missed carry a negative index: don't look up a sphere for them.
    if (is_no_hit(i)) return vec3(1,0,1);

    switch (material_of(i)) {
        case SPHERE_MAT:
        case FLOOR_MAT:
        {
           vec3 n = normal(p);
           RayMarchResult r = ray_march(p + n * 0.01, dl.d, 100, 100.);
           float k = sin(iTime) * 2 + 6;
           // return;
           return vec3(clamp(k * r.min_d, 0, 1));
           // return checker(p);
        }
        // case FLOOR_MAT: return vec3(1);
        // case NO_HIT_DISTANCE_EXHAUST_MAT: return vec3(0,1,1);
        case NO_HIT_DISTANCE_EXHAUST_MAT:
        case NO_HIT_STEP_EXHAUST_MAT: return vec3(1,0,1);
        default: break;
    }
    return vec3(0);
}

void mainImage( out vec4 fragColor, in vec2 fragCoord )
{
    DirectionalLight dl;
    dl.d = normalize(vec3(-5,10,-10)- vec3(0,0,10));
    dl.c = vec3(219. / 255., 192. / 255., 151. / 255.);

    vec2 uv = (fragCoord - 0.5 * iResolution.xy) / iResolution.y;
    vec3 ro = vec3(0, 5,-10);
    vec3 rd = normalize(vec3(uv, 1));

    RayMarchResult r = ray_cast(ro, rd, MAX_DISTANCE);
    vec3 c = color(r.i, dl, r.p);
    // uv *= 10;
    // c = vec3(step(.5, fract(uv.x)));
    c *= 0.05;
    fragColor = vec4(c, 1);
}
