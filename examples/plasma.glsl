// Edit and save: the renderer reloads instantly.
void mainImage(out vec4 fragColor, in vec2 fragCoord)
{
    vec2 uv = (fragCoord - 0.5 * iResolution.xy) / iResolution.y;
    float t = iTime;
    float v = sin(uv.x * 8.0 + t) + sin(uv.y * 8.0 + t * 1.3)
            + sin((uv.x + uv.y) * 6.0 + t * 0.7)
            + sin(length(uv) * 12.0 - t * 2.0);
    vec3 col = 0.5 + 0.5 * cos(v + vec3(0.0, 2.0, 4.0) + t * 0.3);
    fragColor = vec4(col, 1.0);
}
