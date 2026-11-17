// Common: prepended to BOTH sphere.glsl (Image) and sphere.bufferA.glsl.
// Anything the two passes must agree on lives here, once.
//
// The scene is GRID x GRID spheres, one per cell of a square field, plus a floor.
// Spheres are numbered in Morton (Z-order) order of their cell, so any run of
// consecutive indices is a compact square-ish patch of the field. That makes a
// complete binary tree over the indices a bounding-volume hierarchy for free:
// node k covers a fixed index range and holds a bounding sphere of its leaves.

#define GRID_BITS 5                          // field is 2^GRID_BITS cells per side
#define GRID (1 << GRID_BITS)
#define SPHERE_COUNT (GRID * GRID)           // leaves; must stay a power of 4
#define TREE_DEPTH (2 * GRID_BITS)           // levels below the root

#define CELL 2.5                             // cell size; must be >= 2 * max radius
#define FIELD_ORIGIN vec2(-40., -4.)         // xz of the field's min corner

#define SPHERE_MAT 0
#define FLOOR_MAT 1

// Object ids: 0..SPHERE_COUNT-1 are spheres, then the floor.
#define FLOOR_ID SPHERE_COUNT
const vec4 FLOOR_SPHERE = vec4(0., -6e2, 10., 6e2);   // (center.xyz, radius)

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

// Buffer A data block, addressed as a linear texel index t -> (t % DATA_W, t / DATA_W):
//   t in [1, 2N)    BVH node k = t, heap order: root = 1, children of k = 2k, 2k+1,
//                   leaves are k in [N, 2N) (leaf i = k - N, i = sphere index).
//                   texel = bounding sphere (center.xyz, radius); for a leaf that
//                   is exactly the sphere.
//   t in [2N, 3N)   leaf attributes for sphere i = t - 2N: (color.rgb, material)
// The canvas must be at least DATA_W wide and DATA_ROWS tall.
#define DATA_W 128
#define NODE_BASE 0
#define ATTR_BASE (2 * SPHERE_COUNT)
#define DATA_TEXELS (3 * SPHERE_COUNT)

ivec2 texel_pos(int t)
{
    return ivec2(t & (DATA_W - 1), t / DATA_W);
}
