// TLOU RT - the forward relight (v50; tlou_rt.cpp relight). After CS_DeferredLighting the game draws forward layers into the
// light buffer - the eyes (MainPsCheapForward), MainPsStandard - lit by its own probes and lights, not by the path tracing:
// they glowed in rooms the path tracing made darker and went dark where it made them brighter. Right before the game's fog
// (CS_ApplyFogDirectlyToFullRes) the add-on copies the light buffer into Data at g_image and runs this pass on the copy,
// which is copied back: where the layers changed a pixel from what the RE9 renderer left there (P), the change is scaled by
// the path tracing's change of the lighting around it - the luminance of P over the game's own lighting result G, summed
// over the 3 x 3 tiles of 16 x 16 pixels around (re9.hlsl cs_re9_ratio). A layer that brightened the pixel (eyes' wetness,
// a lit cornea) keeps r times its added light; one that darkened it under a brighter path-traced result is raised by r up to P
// (r > 1); otherwise it stays. Runs only on the data of this frame's composite (the token cs_re9_ratio writes).
RWByteAddressBuffer Data : register(u0);
cbuffer relight_constants : register(b0)
{
	uint g_w;                   // the light buffer's size
	uint g_h;
	uint g_pitch;               // the copy's row pitch (bytes) ...
	uint g_image;               // ... and its offset in Data (R11G11B10)
	uint g_gp;                  // per pixel: the game's lighting result G, this renderer's P (R11G11B10 each)
	uint g_tiles;               // 16-byte header (x: the frame token), then per 16 x 16 tile: luminance sums of P, G (float2)
	uint g_frame_info;          // FRAME_INFO in rt.hlsl (x: frames set up so far)
	float g_strength;           // the Relight forward layers setting
};

#define RELIGHT_TILE 16
#define RELIGHT_SKY 0xFFFFFFFFu     // v54: re9.hlsl - a sky pixel: left as the game drew it (its sky comes after the composite)

uint pack_r11g11b10(float3 rgb)
{
	rgb = clamp(rgb, 0.0, asfloat(0x477C0000u));
	const uint r = ((f32tof16(rgb.x) + 8) >> 4) & 0x7FFu;
	const uint g = ((f32tof16(rgb.y) + 8) << 7) & 0x3FF800u;
	const uint b = ((f32tof16(rgb.z) + 16) << 17) & 0xFFC00000u;
	return r | g | b;
}
float3 unpack_r11g11b10(uint v)
{
	return float3(f16tof32((v << 4) & 0x7FF0u), f16tof32((v >> 7) & 0x7FF0u), f16tof32((v >> 17) & 0x7FE0u));
}

float2 tile_sums(int2 t)
{
	const int2 tiles = int2((g_w + RELIGHT_TILE - 1) / RELIGHT_TILE, (g_h + RELIGHT_TILE - 1) / RELIGHT_TILE);
	if (t.x < 0 || t.y < 0 || t.x >= tiles.x || t.y >= tiles.y)
		return 0.0;
	const float2 v = asfloat(Data.Load2(g_tiles + 16 + (t.y * tiles.x + t.x) * 8));
	return all(isfinite(v)) ? max(v, 0.0) : 0.0;
}

[numthreads(8, 8, 1)]
void cs_relight(uint3 id : SV_DispatchThreadID)
{
	if (id.x >= g_w || id.y >= g_h)
		return;
	if (Data.Load(g_tiles) != Data.Load(g_frame_info))
		return;                                             // not this frame's composite: the copy goes back unchanged
	const uint at = g_image + id.y * g_pitch + id.x * 4;
	const uint n_bits = Data.Load(at);
	const uint2 gp = Data.Load2(g_gp + (id.y * g_w + id.x) * 8);
	if (n_bits == gp.y || gp.y == RELIGHT_SKY)
		return;                                             // no forward layer here; the sky (v54)
	const float3 N = unpack_r11g11b10(n_bits), P = unpack_r11g11b10(gp.y);
	const int2 t = int2(id.xy) / RELIGHT_TILE;
	float2 s = 0.0;
	[unroll]
	for (int dy = -1; dy <= 1; ++dy)
		[unroll]
		for (int dx = -1; dx <= 1; ++dx)
			s += tile_sums(t + int2(dx, dy));
	const float r = lerp(1.0, clamp((s.x + 1e-3) / (s.y + 1e-3), 1.0 / 32.0, 8.0), saturate(g_strength));
	const float3 d = N - P;
	float3 o;
	[unroll]
	for (uint c = 0; c < 3; ++c)
		o[c] = d[c] >= 0.0 ? P[c] + d[c] * r : min(N[c] * max(r, 1.0), P[c]);
	Data.Store(at, pack_r11g11b10(max(o, 0.0)));
}
