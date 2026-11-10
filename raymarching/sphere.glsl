#define MAX_STEPS 1000
#define MAX_DISTANCE 1e3

#define NO_HIT_STEP_EXHAUST_MAT -2
#define NO_HIT_DISTANCE_EXHAUST_MAT -1
#define SPHERE_MAT 0
#define FLOOR_MAT 1

struct DirectionalLight
{
    vec3 d;
    vec3 c;
};

struct Sphere
{
    vec3 p;
    float r;
    int m;
    vec3 c;
};

struct RayMarchResult
{
    vec3 p;
    float min_d;
    int i;
};

#define SPHERE_COUNT 10

uint xorshift32(out uint seed) {
	// Algorithm "xor" from p. 4 of Marsaglia, "Xorshift RNGs"
	uint x = seed;
	x ^= x << 13;
	x ^= x >> 17;
	x ^= x << 5;
	return seed = x;
}

float random(float min, float max, out uint seed)
{
    uint value =xorshift32(seed);
    uint max_value = ~0U;
    float t = float(value) / float(max_value);
    return mix(min, max, t);
}

vec3 random_color(out uint seed)
{
    return vec3(random(0,1,seed), random(0,1,seed), random(0,1,seed));        
}

struct Scene
{
    Sphere s[SPHERE_COUNT];
    DirectionalLight dl;
};

float sphere(vec3 p, float r)
{
    return length(p) - r;
}

float map(vec3 p, Scene s, out int i_out)
{
    float min_d = sphere(s.s[0].p - p, s.s[0].r);
    i_out = 0;
    for (int i = 1; i < SPHERE_COUNT; i++)
    {
        float sd = sphere(s.s[i].p - p, s.s[i].r);
        if (min_d > sd)
        {
            min_d = sd;
            i_out = i;
        }
    }
    return min_d;
}

vec3 normal(vec3 p, Scene s)
{
    float e = 1e-3;
    int m;

    return normalize(vec3(
        map(p + vec3(e,0,0), s, m) -
        map(p - vec3(e,0,0), s, m),

        map(p + vec3(0,e,0), s, m) -
        map(p - vec3(0,e,0), s, m),

        map(p + vec3(0,0,e), s, m) -
        map(p - vec3(0,0,e), s, m)
    ));
}

bool is_no_hit(int m)
{
    return m == NO_HIT_DISTANCE_EXHAUST_MAT || m == NO_HIT_STEP_EXHAUST_MAT; 
}

RayMarchResult ray_march(vec3 ro, vec3 rd, Scene s, int max_steps, float max_distance)
{
    vec3 c = vec3(0);
    RayMarchResult r;
    r.p = ro;
    r.min_d = 1e20;
    float t = 0.;
    for (int i = 0; i < max_steps; i++)
    {
        float d = map(r.p, s, r.i);
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

vec3 phong_bling(in Scene s, vec3 p)
{
    float d = max(dot(normal(p,s), s.dl.d), 0.);
    vec3 c = vec3(0.3, 0.5, 0.7);
    return mix(c, s.dl.c ,d);  
}

vec3 checker(vec3 p)
{
    vec2 g = floor(p.xz);
    return vec3(mod(g.x + g.y, 2.0));
}

vec3 color(int m, in Scene s, vec3 p)
{
    switch (m) {
        case SPHERE_MAT:
        case FLOOR_MAT:
        {
           vec3 n = normal(p, s);
           RayMarchResult r = ray_march(p + n * 0.01, s.dl.d, s, 100, 100.);
           float k = sin(iTime) * 2 + 6;
           return 
           // return vec3(clamp(k * r.min_d, 0, 1));
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

void generate_spheres(inout Scene s, out uint seed)
{
    s.s[0] = Sphere(vec3(0, -6e2, 10), 6e2, SPHERE_MAT, random_color(seed));
    vec2 r_range = vec2(0.2, 1.2);
    vec2 area = vec2(-5, 5);
    for (int i = 1; i < SPHERE_COUNT; i++)
    {
        for(int tries = 0; tries < 50; tries++)
        {
            float r = random(r_range.x, r_range.y, seed);
            vec3 p = vec3(random(area.x, area.y, seed), r, random(area.x, area.y, seed));
            bool free = true;
            for(int j = 0; j < i; j++)
            {
                if (distance(s.s[i].p, s.s[j].p) - s.s[i].r - s.s[j].r < 0)
                {
                    free = false;
                    break;
                }
            }

            if (free)
            {
                s.s[i] = Sphere(p, r, SPHERE_MAT, random_color(seed));
                break;        
            }
        }            
    }

    // s.s[1] = Sphere(vec3(2, 1, 2), 1, SPHERE_MAT);
    // s.s[2] = Sphere(vec3(0.5, 0.2, 3), 0.2, SPHERE_MAT);
    // s.s[3] = Sphere(vec3(-5, 0.3, 5), 0.3, SPHERE_MAT);
    // s.s[4] = Sphere(vec3(5, 0.7, 10), 0.7, SPHERE_MAT);
    // s.s[5] = Sphere(vec3(0, 1, 2), 1, SPHERE_MAT);
    // s.s[6] = Sphere(vec3(0, 0.2, 3), 0.2, SPHERE_MAT);
    // s.s[7] = Sphere(vec3(0, 0.3, 5), 0.3, SPHERE_MAT);       
}

void mainImage( out vec4 fragColor, in vec2 fragCoord )
{
    uint seed = 1337;
    Scene s;
    generate_spheres(s, seed);

    s.dl.d = normalize(vec3(-5,10,-10)- vec3(0,0,10));
    s.dl.c = vec3(219. / 255., 192. / 255., 151. / 255.);

    vec2 uv = (fragCoord - 0.5 * iResolution.xy) / iResolution.y;
    vec3 ro = vec3(0, 5,-10);
    vec3 rd = normalize(vec3(uv, 1));

    RayMarchResult r = ray_march(ro, rd, s, MAX_STEPS, MAX_DISTANCE);
    vec3 c = color(r.m, s, r.p);
    // uv *= 10;
    // c = vec3(step(.5, fract(uv.x)));
    c *= 0.08;
    fragColor = vec4(c, 1);
}
