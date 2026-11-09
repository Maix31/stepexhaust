#define MAX_STEPS 100
#define MAX_DISTANCE 1e3

#define SPHERE_MAT 0
#define FLOOR_MAT 1
#define NO_HIT_DISTANCE_EXHAUST_MAT 2
#define NO_HIT_STEP_EXHAUST_MAT 3

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
};

struct RayMarchResult
{
    vec3 p;
    int m;
};

#define sr 2.0
#define fr 6e2

struct Scene
{
    Sphere s;
    Sphere f;
    DirectionalLight dl;
};

float sphere(vec3 p, float r)
{
    return length(p) - r;
}

float map(vec3 p, Scene s, out int m)
{
    float sd = sphere(s.s.p - p, s.s.r);
    float fd = sphere(s.f.p - p, s.f.r);
    if (sd < fd)
    {
        m = s.s.m;
        return sd;
    } else
    {
        m = s.f.m;
        return fd;
    }
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

RayMarchResult ray_march(vec3 ro, vec3 rd, Scene s)
{
    vec3 c = vec3(0);
    RayMarchResult r;
    r.p = ro;
    float t = 0.;
    for (int i = 0; i < MAX_STEPS; i++)
    {
        float d = map(r.p, s, r.m);
        float eps = max(1e-4, 0.5 * t / iResolution.y);

        if (d < eps)
        {
            return r;
        }
        
        if (t > MAX_DISTANCE)
        {
            r.m = NO_HIT_DISTANCE_EXHAUST_MAT;
            return r;
        }

        r.p += rd * d;
        t += d;
    }

    r.m = NO_HIT_STEP_EXHAUST_MAT;
    return r;
}

vec3 phong_bling(in Scene s, vec3 p)
{
    float d = max(dot(normal(p,s), s.dl.d), 0.);
    vec3 c = vec3(0.3, 0.5, 0.7);
    return mix(c, s.dl.c ,d);  
}

vec3 color(int m, in Scene s, vec3 p)
{
    switch (m) {
        case SPHERE_MAT: {
           return phong_bling(s, p);
        }
        // case FLOOR_MAT: return vec3(1);
        case FLOOR_MAT:
        {
            vec2 g = floor(p.xz);
            return vec3(mod(g.x + g.y, 2.0));
        }
        // case NO_HIT_DISTANCE_EXHAUST_MAT: return vec3(0,1,1);
        case NO_HIT_DISTANCE_EXHAUST_MAT:
        case NO_HIT_STEP_EXHAUST_MAT: return vec3(1,0,1);
        default: break;
    }
    return vec3(0);
}

void mainImage( out vec4 fragColor, in vec2 fragCoord )
{
    Scene s;
    s.s = Sphere(vec3(0, sr, 10), sr, SPHERE_MAT);
    s.f = Sphere(vec3(0, -fr, 10), fr, FLOOR_MAT);
    s.dl.d = normalize(vec3(-5,10,-10)- vec3(0,0,10));
    s.dl.c = vec3(219. / 255., 192. / 255., 151. / 255.);

    vec2 uv = (fragCoord - 0.5 * iResolution.xy) / iResolution.y;
    vec3 ro = vec3(0, 5,-10);
    vec3 rd = normalize(vec3(uv, 1));

    RayMarchResult r = ray_march(ro, rd, s);
    vec3 c = color(r.m, s, r.p);
    // uv *= 10;
    // c = vec3(step(.5, fract(uv.x)));
    c *= 0.08;
    fragColor = vec4(c, 1);
}
