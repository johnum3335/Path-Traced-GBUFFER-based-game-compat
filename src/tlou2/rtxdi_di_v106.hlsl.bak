// TLOU RT - NVIDIA's RTXDI passes (v76: ReSTIR DI for the lamps; user: "try using option A, use NVIDIAS library"). NVIDIA's
// RTXDI library is compiled in unmodified (RTGI_builds\deps\rtxdi-library, under the NVIDIA RTX SDKs license: it ships in
// object code only - "This software contains source code provided by NVIDIA Corporation."), on the add-on's data through its
// bridge (rtxdi_bridge.hlsli); its switches and resources: rtxdi.hlsli (v77, through re9.hlsl).
// cs_rtxdi_di - one fused pass per pixel, as NVIDIA's minimal sample: the lamps drawn (RTXDI_SampleLocalLights - v77 from the
// pixel's ReGIR cell, else power-proportional, else uniform: g_rtxdi.initial), the pick's shadow ray (a blocked pick leaves
// the reservoir, its M stays), RTXDI_DISpatioTemporalResampling (last frame's reservoir and the neighbours', the bias
// correction of the "RTXDI bias correction" setting), then the final pick shaded with its own shadow ray - a lamp the pixel
// does not see is dropped from the reservoir before it is stored, so neither the next frame nor the neighbours take it
// (v72-v75's trouble with the add-on's own ReSTIR DI: a lamp shadowed there kept coming back). The result goes to RE9_RTXDI
// in di_shade's units; cs_re9_path (v77: and cs_dd2_path's primary lamps) take it.
// v77: cs_rtxdi_pdf, cs_rtxdi_presample, cs_regir_build - NVIDIA's presampling and ReGIR for the lamps, run every frame after
// the lamps' list (tlou_rt.cpp post_lighting); the paths' bounce hits sample their ReGIR cell (rtxdi.hlsli rtxdi_lamps_at).
// v86: cs_rtxdi_game - with "NVIDIA RTXDI game lights" the game's own lights join the lamps in RTXDI's list (its power PDF,
// RIS tiles, ReGIR cells, ReSTIR DI): one estimate for both at the camera's surfaces (cs_rtxdi_di, its specular part apart)
// and at the bounce hits, in place of the per-light loops and the hits' uniform pick of v71-v85.
#include "re9.hlsl"

// RTXDI's screen-space motion: where the pixel's surface was last frame, in pixels (the triangle's own motion - cs_motion -
// or the camera's), and the change of its view depth
float3 rtxdi_motion(uint2 px, RAB_Surface surface)
{
	const float3 clip = previous_clip(px, surface.worldPos);
	if (!(clip.z > 0.0))
		return float3(1e5, 1e5, 0.0);                   // not on screen last frame: no temporal reuse
	const float2 fpos = float2(clip.x * 0.5 + 0.5, 0.5 - clip.y * 0.5) * float2(full_size()) - 0.5;
	return float3(fpos - float2(px), clip.z - surface.viewDepth);
}

[numthreads(8, 8, 1)]
void cs_rtxdi_di(uint3 id : SV_DispatchThreadID)
{
	const uint2 size = full_size();
	if (id.x >= size.x || id.y >= size.y)
		return;
	const RAB_Surface surface = RAB_GetGBufferSurface(int2(id.xy), false);
	RTXDI_DIReservoir reservoir = RTXDI_EmptyDIReservoir();
	float3 light = 0.0, light_s = 0.0;                     // v86: the diffuse and the specular part apart
	if (RAB_IsSurfaceValid(surface))
	{
		RTXDI_RandomSamplerState rng = RTXDI_InitRandomSampler(id.xy, g_rtxdi.runtime.frameIndex, 1u);
		// RTXDI's "coherent" stream: the ReGIR cell's jitter (up to half a metre and more around the onion's centre) and the RIS
		// tile. v85 (user: "Regir looks broken" - bright 16 x 16 pixel squares of lamp light, a frame each, on the 16-pixel grid
		// in v84's capture): per pixel. v76-v84 drew it per 16 x 16 tile (RTXDI_TILE_SIZE_IN_PIXELS, NVIDIA's sample's cache
		// coherence): a tile whose jittered cell held a lamp rarely got that lamp's large weight in all its pixels at once
		RTXDI_RandomSamplerState tile_rng = RTXDI_InitRandomSampler(id.xy, g_rtxdi.runtime.frameIndex, 2u);
		RTXDI_LightBufferRegion lamps = (RTXDI_LightBufferRegion)0;
		lamps.numLights = rtxdi_light_count(false);
		RAB_LightSample sample = RAB_EmptyLightSample();
		reservoir = RTXDI_SampleLocalLights(rng, tile_rng, surface, g_rtxdi.initial, RTXDI_ComputeInitialSamplingMisData(g_rtxdi.initial),
			g_rtxdi.initial.localLightSamplingMode, lamps, g_rtxdi.localRis, g_rtxdi.regir, sample);
		if (RTXDI_IsValidDIReservoir(reservoir) && !RAB_GetConservativeVisibility(surface, sample))
			RTXDI_StoreVisibilityInDIReservoir(reservoir, 0.0, true);
		if (g_rtxdi.reuse != 0u)
		{
			int2 temporal_px;
			reservoir = RTXDI_DISpatioTemporalResampling(id.xy, surface, reservoir, rng, rtxdi_motion(id.xy, surface), g_rtxdi.input,
				g_rtxdi.runtime, g_rtxdi.reservoirs, g_rtxdi.spatiotemporal, temporal_px, sample);
		}
		if (RTXDI_IsValidDIReservoir(reservoir))
		{
			if (RAB_GetConservativeVisibility(surface, sample))
			{
				rtxdi_reflected_split(sample, surface, light, light_s);
				light *= RTXDI_GetDIReservoirInvPdf(reservoir);
				light_s *= RTXDI_GetDIReservoirInvPdf(reservoir);
			}
			else
				RTXDI_StoreVisibilityInDIReservoir(reservoir, 0.0, true);
		}
		// di_shade's units (before the lighting's factor) and its clamp (both parts together)
		const float3 post = max(lighting_factor(id.xy, Depth[id.xy]) * g_brightness, 1e-4);
		light /= post;
		light_s /= post;
		if (!all(isfinite(light)) || !all(isfinite(light_s)))
		{
			light = 0.0;
			light_s = 0.0;
		}
		const float le = restir_lum(light + light_s);
		if (le > DI_MAX)
		{
			light *= DI_MAX / le;
			light_s *= DI_MAX / le;
		}
	}
	RTXDI_StoreDIReservoir(reservoir, g_rtxdi.reservoirs, id.xy, g_rtxdi.output);
	re9_store(RE9_RTXDI, id.xy, float4(max(light, 0.0), asfloat(pack_r11g11b10(max(light_s, 0.0)))));   // v86: w the specular
}

// v77: the lamps' power PDF for RTXDI's presampling - lamp i of this frame's list at its Z-curve texel of a pdfSize x pdfSize
// mip 0 (RTXDI_LinearIndexToZCurve, as RTXDI_PresampleLocalLights maps a texel back to a lamp: its luminance x area), and the
// mip chain to 1 x 1 (each texel the mean of the 2 x 2 below, as RTXDI asks). One group of 32 x 32 threads (EMIT_MAX lamps)
groupshared float g_pdf_a[1024];
groupshared float g_pdf_b[1024];
[numthreads(32, 32, 1)]
void cs_rtxdi_pdf(uint3 gt : SV_GroupThreadID, uint gi : SV_GroupIndex)
{
	const uint size = g_rtxdi.pdfSize;
	const uint2 texel = RTXDI_LinearIndexToZCurve(gi);
	float power = 0.0;
	if (gi < rtxdi_light_count(false))
	{
		const RAB_LightInfo light = RAB_LoadLightInfo(gi, false);
		if (light.game != 0u)
		{
			// v86: a game's light - its colour x Brightness x the square of the distance its falloff halves at (the lamps:
			// radiance x area x Lamps; the two weigh against each other as they light)
			const LightDesc L = Lights[light.game - 1u];
			const float x = max(L.falloff.x, 0.0), y = max(L.falloff.y, 0.0);
			const float half_d = y > 1e-6 ? (sqrt(x * x + 4.0 * y) - x) / (2.0 * y) : (x > 1e-6 ? 1.0 / x : 10.0);
			power = restir_lum(light.Le) * g_brightness * max(L.falloff.z, 0.0) * clamp(half_d, 0.1, 50.0) * clamp(half_d, 0.1, 50.0);
		}
		else
		{
			power = restir_lum(light.Le) * light.area * g_emit;
			// v105: a lamp the camera's ray did not reach this frame (rt.hlsl cs_lamp_merge: LAMP_HIDDEN) weighs less - the
			// lamps' samples go to the lamps in view or open to it (never 0: its light still comes)
			if (light.lamp != 0u && (Data.Load(lamp_offset() + (light.lamp - 1u) * LAMP_BYTES + 44u) & LAMP_HIDDEN) != 0u)
				power *= LAMP_HIDDEN_WEIGHT;
		}
		if (!isfinite(power))
			power = 0.0;
	}
	if (texel.x < size && texel.y < size)
	{
		g_pdf_a[texel.y * size + texel.x] = max(power, 0.0);
		Data.Store(g_rtxdi.pdfAt + (texel.y * size + texel.x) * 4u, asuint(max(power, 0.0)));
	}
	GroupMemoryBarrierWithGroupSync();
	uint at = g_rtxdi.pdfAt + size * size * 4u;
	[loop]
	for (uint m = 1u; (size >> m) >= 1u; ++m)
	{
		const uint s = size >> m, below = s * 2u;
		float v = 0.0;
		if (gt.x < s && gt.y < s)
		{
			const uint b = (gt.y * 2u) * below + gt.x * 2u;
			v = (m & 1u) != 0u ? g_pdf_a[b] + g_pdf_a[b + 1u] + g_pdf_a[b + below] + g_pdf_a[b + below + 1u]
				: g_pdf_b[b] + g_pdf_b[b + 1u] + g_pdf_b[b + below] + g_pdf_b[b + below + 1u];
			v *= 0.25;
			Data.Store(at + (gt.y * s + gt.x) * 4u, asuint(v));
		}
		GroupMemoryBarrierWithGroupSync();
		if (gt.x < s && gt.y < s)
		{
			if ((m & 1u) != 0u)
				g_pdf_b[gt.y * s + gt.x] = v;
			else
				g_pdf_a[gt.y * s + gt.x] = v;
		}
		GroupMemoryBarrierWithGroupSync();
		at += s * s * 4u;
	}
}

// v77: RTXDI's RIS tiles of lamps (localRis.tileCount tiles of localRis.tileSize), each entry a lamp drawn in proportion to
// its power - the candidates of ReSTIR DI's power mode and ReGIR's fallback outside the onion
[numthreads(256, 1, 1)]
void cs_rtxdi_presample(uint3 id : SV_DispatchThreadID)
{
	if (id.x >= g_rtxdi.localRis.tileSize || id.y >= g_rtxdi.localRis.tileCount)
		return;
	RTXDI_RandomSamplerState rng = RTXDI_InitRandomSampler(id.xy, g_rtxdi.runtime.frameIndex, 0u);
	rtxdi_pdf_texture pdf;
	pdf.unused = 0u;
	RTXDI_LightBufferRegion lamps = (RTXDI_LightBufferRegion)0;
	lamps.numLights = rtxdi_light_count(false);
	RTXDI_PresampleLocalLights(rng, pdf, uint2(g_rtxdi.pdfSize, g_rtxdi.pdfSize), id.y, id.x, lamps, g_rtxdi.localRis);
}

// v92 (user, v91: "lowering the value to 2 removes the noise, 3 makes it come back"): a ReGIR slot filled as
// RTXDI_PresampleLocalLightsForReGIR fills it (RTXDI's light selection, its slot format: RTXDI's sampling reads it unchanged),
// with the candidates per slot: one for a cell's first g_rtxdi.regirPlainSlots slots - a power-proportional pick, every light
// that lights the cell in reach (the ReGIR article's canonical samples) -, numRegirBuildSamples for the rest. The cells'
// weights know no shadows: with many candidates the strongest light of a cell takes nearly every slot (each candidate makes it
// likelier: 1 - 0.6^M), and where it is shadowed the others came rarely, with large weights - v88-v91's bright noise. Each
// slot's weight stays its own 1 / pdf: the pixels' RIS over random slots stays unbiased
void regir_build_mixed(inout RTXDI_RandomSamplerState rng, inout RTXDI_RandomSamplerState coherentRng, uint lightSlot,
	RTXDI_LightBufferRegion localLightBufferRegion, RTXDI_RISBufferSegmentParameters localLightRISBufferSegmentParams, ReGIR_Parameters regirParams)
{
	const uint risBufferPtr = regirParams.commonParams.risBufferOffset + lightSlot;
	const uint inCell = lightSlot % regirParams.commonParams.lightsPerCell;
	const uint m = inCell < g_rtxdi.regirPlainSlots ? 1u : regirParams.commonParams.numRegirBuildSamples;
	float3 cellCenter;
	float cellRadius;
	if (m == 0u || !RTXDI_ReGIR_CellIndexToWorldPos(regirParams, int(lightSlot / regirParams.commonParams.lightsPerCell), cellCenter, cellRadius))
	{
		RTXDI_RIS_BUFFER[risBufferPtr] = uint2(0, 0);
		return;
	}
	// every point whose jittered lookup can land in this cell: the cell's radius + the lookup jitter's reach at the cell's far
	// side (RTXDI_ReGIR_GetJitterScale: the edge of a cube around the point - its half diagonal). RTXDI's (samplingJitter + 1)
	// x the radius falls short of it (3.2 m of jitter against 2.1 m of widening at 5 m, 9 m against 5 m at 20 m): fine for its
	// distance-only weight, but v88's cone and cardioid terms then gave a pixel in a light's beam a cell that weighed the
	// light near 0 - with 2 candidates and more, a huge weight whenever it was picked (v88-v91's bright noise)
	const float centreDistance = length(cellCenter);
	const float3 far_point = centreDistance > 1e-3 ? cellCenter * ((centreDistance + cellRadius) / centreDistance) : float3(cellRadius, 0.0, 0.0);
	cellRadius += 0.8660254 * RTXDI_ReGIR_GetJitterScale(regirParams, far_point);
	RTXDI_LocalLightSelectionContext ctx;
	const bool power = regirParams.commonParams.localLightPresamplingMode == REGIR_LOCAL_LIGHT_PRESAMPLING_MODE_POWER_RIS;
	if (power)
		ctx = RTXDI_InitializeLocalLightSelectionContextRIS(coherentRng, localLightRISBufferSegmentParams);
	else
		ctx = RTXDI_InitializeLocalLightSelectionContextUniform(localLightBufferRegion);
	regir_source source = regir_source_init(ctx, power);  // v96: + the game's lights alike (rtxdi.hlsli)
	uint selected = 0u;
	float selectedTarget = 0.0, sum = 0.0;
	[loop]
	for (uint i = 0u; i < m; ++i)
	{
		uint light;
		RAB_LightInfo info = RAB_EmptyLightInfo();
		float invSourcePdf;
		regir_source_next(source, rng, light, info, invSourcePdf);
		const float target = RAB_GetLightTargetPdfForVolume(info, cellCenter, cellRadius);
		const float w = target * invSourcePdf / float(m);
		sum += w;
		if (RTXDI_GetNextRandom(rng) * sum < w)
		{
			selected = light;
			selectedTarget = target;
		}
	}
	const float weight = selectedTarget > 0.0 ? sum / selectedTarget : 0.0;
	RTXDI_RIS_BUFFER[risBufferPtr] = uint2(selected, asuint(weight));   // (RAB_StoreCompactLightInfo: none - no compact bit)
}

// v77: ReGIR's cells around the camera - each light slot of each cell a lamp picked by RIS over power-proportional candidates,
// weighted for the cell (RAB_GetLightTargetPdfForVolume). v92: regir_build_mixed (plain and targeted slots)
[numthreads(256, 1, 1)]
void cs_regir_build(uint3 id : SV_DispatchThreadID)
{
	const uint slot = id.x;
	if (slot >= g_rtxdi.regirSlots)
		return;
	RTXDI_RandomSamplerState rng = RTXDI_InitRandomSampler(uint2(slot & 0xFFFu, slot >> 12), g_rtxdi.runtime.frameIndex, 1u);
	RTXDI_RandomSamplerState cell_rng = RTXDI_InitRandomSampler(uint2(slot >> 8, 0u), g_rtxdi.runtime.frameIndex, 1u);
	RTXDI_LightBufferRegion lamps = (RTXDI_LightBufferRegion)0;
	lamps.numLights = rtxdi_light_count(false);
	regir_build_mixed(rng, cell_rng, slot, lamps, g_rtxdi.localRis, g_rtxdi.regir);
}

// v86 (user: "go for rtxdi for sky and game lights"): this frame's game lights for RTXDI, after the lamps in its list - the
// census' list (cs_light_census, before the game's lighting; none while the game lights them itself) with each light's
// identity (its index in the game's records, its flags, radius and cookie) and the map from a record's index to its slot +
// 1, for last frame's samples to find their light again (rtxdi_bridge.hlsli RAB_TranslateLightIndex): the census lists the
// lights in a new order every frame. One group, before the lamps' PDF (tlou_rt.cpp post_lighting)
[numthreads(1024, 1, 1)]
void cs_rtxdi_game(uint3 t : SV_GroupThreadID)
{
	const uint at = rtxdi_game_at(false);
	Data.Store(at + RTXDI_GAME_MAP + t.x * 4u, 0u);
	DeviceMemoryBarrierWithGroupSync();
	const uint count = g_lights_on != 0 ? min(Data.Load(LIGHT_LIST_COUNT), LIGHT_LIST_MAX) : 0u;
	if (t.x < count)
	{
		const uint index = Data.Load(LIGHT_LIST + t.x * 4u);
		const LightDesc L = Lights[index];
		Data.Store4(at + RTXDI_GAME_SLOTS + t.x * 16u, uint4(index, L.flags, asuint(L.radius), L.cookie));
		Data.Store(at + RTXDI_GAME_MAP + min(index, 1023u) * 4u, t.x + 1u);
	}
	if (t.x == 0u)
		Data.Store2(at, uint2(count, g_frame));
}
