// Buffer A: builds the sphere field and its bounding-volume hierarchy ONCE
// (iFrame == 0) into texels. Later frames just copy themselves forward.
//
// Every texel of the data block is computed independently (no pass depends on
// another), so the whole tree is built in a single frame. Sphere i is a pure
// function of i, so a node simply regenerates the leaves it covers.
//
// Layout, constants and Sphere come from sphere.common.glsl

#define R_MIN 0.2
#define R_MAX 1.2   // CELL must be >= 2 * R_MAX so a sphere never leaves its cell

uint hash(uint x)
{
    // "lowbias32" integer hash
    x ^= x >> 16;
    x *= 0x7feb352du;
    x ^= x >> 15;
    x *= 0x846ca68bu;
    x ^= x >> 16;
    return x;
}

uint xorshift32(inout uint seed) {
	// Algorithm "xor" from p. 4 of Marsaglia, "Xorshift RNGs"
	uint x = seed;
	x ^= x << 13;
	x ^= x >> 17;
	x ^= x << 5;
	return seed = x;
}

float random(float min, float max, inout uint seed)
{
    uint value = xorshift32(seed);
    uint max_value = ~0U;
    float t = float(value) / float(max_value);
    return mix(min, max, t);
}

vec3 random_color(inout uint seed)
{
    return vec3(random(0.,1.,seed), random(0.,1.,seed), random(0.,1.,seed));
}

// Gather every second bit of x (inverse of Morton interleaving).
uint compact_bits(uint x)
{
    x &= 0x55555555u;
    x = (x | (x >> 1)) & 0x33333333u;
    x = (x | (x >> 2)) & 0x0f0f0f0fu;
    x = (x | (x >> 4)) & 0x00ff00ffu;
    x = (x | (x >> 8)) & 0x0000ffffu;
    return x;
}

// Sphere i lives in cell Morton-decode(i) and is placed fully inside it, so
// spheres never overlap and need no rejection sampling.
Sphere make_sphere(int i)
{
    uint seed = hash(uint(i) + 69u);
    ivec2 cell = ivec2(compact_bits(uint(i)), compact_bits(uint(i) >> 1));
    float r = random(R_MIN, R_MAX, seed);
    float x = random(r, CELL - r, seed);
    float z = random(r, CELL - r, seed);
    vec3 p = vec3(FIELD_ORIGIN.x + float(cell.x) * CELL + x, r,
                  FIELD_ORIGIN.y + float(cell.y) * CELL + z);
    return Sphere(p, r, SPHERE_MAT, random_color(seed));
}

// Bounding sphere of BVH node k: (center.xyz, radius).
vec4 build_bound(int k)
{
    if (k >= SPHERE_COUNT)
    {
        Sphere s = make_sphere(k - SPHERE_COUNT);
        return vec4(s.p, s.r);
    }

    int depth = 0;
    while ((k >> (depth + 1)) > 0) depth++;
    int size = SPHERE_COUNT >> depth;          // leaves under this node
    int lo = (k - (1 << depth)) * size;        // first leaf index

    vec3 mn = vec3(1e9);
    vec3 mx = vec3(-1e9);
    for (int j = 0; j < size; j++)
    {
        Sphere s = make_sphere(lo + j);
        mn = min(mn, s.p - s.r);
        mx = max(mx, s.p + s.r);
    }
    vec3 c = 0.5 * (mn + mx);
    float R = 0.;
    for (int j = 0; j < size; j++)
    {
        Sphere s = make_sphere(lo + j);
        R = max(R, distance(c, s.p) + s.r);
    }
    return vec4(c, R);
}

void mainImage(out vec4 fragColor, in vec2 fragCoord)
{
    ivec2 px = ivec2(fragCoord);

    // After frame 0: keep what we have (iChannel0 = this buffer's previous frame).
    if (iFrame > 0) { fragColor = texelFetch(iChannel0, px, 0); return; }

    // Pixels outside the data block do nothing.
    int t = px.x + px.y * DATA_W;
    if (px.x >= DATA_W || t < 1 || t >= DATA_TEXELS) { fragColor = vec4(0); return; }

    if (t < ATTR_BASE)
    {
        fragColor = build_bound(t);
    }
    else
    {
        Sphere s = make_sphere(t - ATTR_BASE);
        fragColor = vec4(s.c, float(s.m));
    }
}
