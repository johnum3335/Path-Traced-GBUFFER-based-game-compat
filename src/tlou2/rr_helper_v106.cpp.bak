// TLOU RT - tlou-rt-rr.exe, the Ray Reconstruction helper (v61; protocol: rr_shared.h). Started by the add-on with the game's
// process id; runs DLSS Ray Reconstruction (NVIDIA NGX, DLSS SDK 310.9.1, nvngx_dlssd.dll next to this exe) on the game's GPU
// in a process of its own: inside the game, RR's creation hung RenoDX's DLSS Neural Rendering and the game (v59, v60).
// Opens the add-on's shared memory, fences and textures by name, creates RR at the render size (DLAA: input = output) and
// evaluates every published frame on its own queue: GPU wait on the input fence, RR, signal of the output fence. Logs to
// tlou-rt-rr.log next to it. Ends with the game (its job, and it watches the game's process).
// v78 (protocol v2): every published frame in order (v61-v77 the newest, older ones skipped - the game's queue now waits for
// each frame's output); before recording RR the queue copies the slot's frame constants (rr_gpu_frame: the jitter the add-on's
// GPU derived from the game's matrices, a reset at a cut) into a readback buffer after the input fence, and the CPU reads
// them; RR is created with MVJittered (the add-on's motion vectors run between the jittered frames); the specular hit
// distance is not passed yet (nor real matrices: identities).
#define NOMINMAX
#include <windows.h>
#include <d3d12.h>
#include <dxgi1_6.h>
#include <algorithm>
#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <iterator>
#include <string>
#include <nvsdk_ngx_helpers_dlssd_d3d.h>
#include "rr_shared.h"

namespace
{
	constexpr const char *k_project = "5d3f9a1c-7b2e-4c80-9e41-2a6b8d0c4f17";
	FILE *s_log = nullptr;
	rr_shared *s_shared = nullptr;

	void say(const char *format, ...)
	{
		char text[512];
		va_list args;
		va_start(args, format);
		std::vsnprintf(text, sizeof(text), format, args);
		va_end(args);
		if (s_log != nullptr)
		{
			SYSTEMTIME t;
			GetLocalTime(&t);
			std::fprintf(s_log, "%02u:%02u:%02u.%03u %s\n", t.wHour, t.wMinute, t.wSecond, t.wMilliseconds, text);
			std::fflush(s_log);
		}
	}

	// the status the add-on shows
	void status(const char *format, ...)
	{
		char text[sizeof(s_shared->message)];
		va_list args;
		va_start(args, format);
		std::vsnprintf(text, sizeof(text), format, args);
		va_end(args);
		say("status: %s", text);
		if (s_shared != nullptr)
		{
			std::memcpy(const_cast<char *>(s_shared->message), text, sizeof(text));
			MemoryBarrier();
		}
	}

	int failed(NVSDK_NGX_Result result, const char *format, ...)
	{
		char text[200];
		va_list args;
		va_start(args, format);
		std::vsnprintf(text, sizeof(text), format, args);
		va_end(args);
		if (s_shared != nullptr)
		{
			InterlockedExchange(&s_shared->helper_result, static_cast<LONG>(result));
			InterlockedExchange(&s_shared->helper_state, rr_state_failed);
		}
		status("%s", text);
		return 1;
	}

	void NVSDK_CONV ngx_log(const char *message, NVSDK_NGX_Logging_Level, NVSDK_NGX_Feature)
	{
		std::string line = message != nullptr ? message : "";
		while (!line.empty() && (line.back() == '\n' || line.back() == '\r'))
			line.pop_back();
		say("NGX: %s", line.c_str());
	}

	// the queue's work so far done (a private fence)
	bool gpu_idle(ID3D12CommandQueue *queue, ID3D12Fence *fence, UINT64 &value, HANDLE event, DWORD timeout)
	{
		++value;
		if (FAILED(queue->Signal(fence, value)) || FAILED(fence->SetEventOnCompletion(value, event)))
			return false;
		return WaitForSingleObject(event, timeout) == WAIT_OBJECT_0;
	}

	void barrier(ID3D12GraphicsCommandList *cl, ID3D12Resource *resource, D3D12_RESOURCE_STATES before, D3D12_RESOURCE_STATES after)
	{
		D3D12_RESOURCE_BARRIER b = {};
		b.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
		b.Transition.pResource = resource;
		b.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
		b.Transition.StateBefore = before;
		b.Transition.StateAfter = after;
		cl->ResourceBarrier(1, &b);
	}
}

int wmain(int argc, wchar_t **argv)
{
	wchar_t exe[MAX_PATH] = {};
	GetModuleFileNameW(nullptr, exe, MAX_PATH);
	std::wstring folder = exe;
	folder.erase(folder.find_last_of(L"\\/"));
	_wfopen_s(&s_log, (folder + L"\\tlou-rt-rr.log").c_str(), L"w");
	if (argc < 2)
		return say("usage: tlou-rt-rr.exe <game process id>"), 2;
	const DWORD pid = std::wcstoul(argv[1], nullptr, 10);
	say("TLOU RT Ray Reconstruction helper for process %lu", static_cast<unsigned long>(pid));
	const HANDLE game = OpenProcess(SYNCHRONIZE, FALSE, pid);
	wchar_t name[160];
	rr_name(name, 160, pid, L"shared");
	const HANDLE mapping = OpenFileMappingW(FILE_MAP_ALL_ACCESS, FALSE, name);
	s_shared = mapping != nullptr ? static_cast<rr_shared *>(MapViewOfFile(mapping, FILE_MAP_ALL_ACCESS, 0, 0, sizeof(rr_shared))) : nullptr;
	if (game == nullptr || s_shared == nullptr || s_shared->magic != TLOU_RR_MAGIC || s_shared->version != TLOU_RR_VERSION)
		return say("the game's shared memory was not found (or another version)"), 3;
	rr_name(name, 160, pid, L"frame");
	const HANDLE frame_event = OpenEventW(SYNCHRONIZE, FALSE, name);
	status("starting");

	// the game's adapter
	IDXGIFactory4 *factory = nullptr;
	IDXGIAdapter1 *adapter = nullptr;
	ID3D12Device *device = nullptr;
	if (FAILED(CreateDXGIFactory2(0, IID_PPV_ARGS(&factory))) || FAILED(factory->EnumAdapterByLuid(s_shared->adapter, IID_PPV_ARGS(&adapter))) ||
		FAILED(D3D12CreateDevice(adapter, D3D_FEATURE_LEVEL_12_0, IID_PPV_ARGS(&device))))
		return failed(NVSDK_NGX_Result_Fail, "no D3D12 device on the game's adapter");
	InterlockedExchange(&s_shared->helper_state, rr_state_device);
	ID3D12CommandQueue *queue = nullptr;
	ID3D12Fence *own_fence = nullptr, *fence_in = nullptr, *fence_out = nullptr;
	// v62: high priority - next to the game (its neural models: RenoDX's Neural Rendering, the optimizer) RR took ~180 ms a
	// frame on a normal queue, 1.6 ms alone
	const D3D12_COMMAND_QUEUE_DESC queue_desc = { D3D12_COMMAND_LIST_TYPE_DIRECT, D3D12_COMMAND_QUEUE_PRIORITY_HIGH };
	device->CreateCommandQueue(&queue_desc, IID_PPV_ARGS(&queue));
	device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&own_fence));
	ID3D12CommandAllocator *allocators[TLOU_RR_SLOTS] = {};
	ID3D12GraphicsCommandList *cl = nullptr, *copy_list = nullptr;   // v78: copy_list - the frame constants' copy
	for (ID3D12CommandAllocator *&a : allocators)
		device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&a));
	if (queue == nullptr || own_fence == nullptr || allocators[TLOU_RR_SLOTS - 1] == nullptr ||
		FAILED(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, allocators[0], nullptr, IID_PPV_ARGS(&cl))) ||
		FAILED(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, allocators[1], nullptr, IID_PPV_ARGS(&copy_list))))   // (one recording list per allocator)
		return failed(NVSDK_NGX_Result_Fail, "its command queue could not be created");
	cl->Close();
	copy_list->Close();
	// v78: the slots' frame constants arrive here (a readback buffer, mapped for good)
	ID3D12Resource *readback = nullptr;
	const uint8_t *readback_ptr = nullptr;
	{
		D3D12_HEAP_PROPERTIES heap = { D3D12_HEAP_TYPE_READBACK };
		D3D12_RESOURCE_DESC desc = {};
		desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
		desc.Width = UINT64(TLOU_RR_SLOTS) * TLOU_RR_CONSTANT_BYTES;
		desc.Height = 1;
		desc.DepthOrArraySize = 1;
		desc.MipLevels = 1;
		desc.SampleDesc.Count = 1;
		desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
		void *mapped = nullptr;
		if (FAILED(device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc, D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&readback))) ||
			FAILED(readback->Map(0, nullptr, &mapped)))
			return failed(NVSDK_NGX_Result_Fail, "its readback buffer could not be created");
		readback_ptr = static_cast<const uint8_t *>(mapped);
	}
	for (const wchar_t *what : { L"fence-in", L"fence-out" })
	{
		rr_name(name, 160, pid, what);
		HANDLE handle = nullptr;
		ID3D12Fence *&fence = what[6] == L'i' ? fence_in : fence_out;
		if (FAILED(device->OpenSharedHandleByName(name, GENERIC_ALL, &handle)) || FAILED(device->OpenSharedHandle(handle, IID_PPV_ARGS(&fence))))
			return failed(NVSDK_NGX_Result_Fail, "the shared fences could not be opened");
		CloseHandle(handle);
	}
	const HANDLE wait_event = CreateEventW(nullptr, FALSE, FALSE, nullptr);
	UINT64 own_value = 0;

	// NGX and RR's availability
	const std::wstring data = folder + L"\\ngx";
	CreateDirectoryW(data.c_str(), nullptr);
	const wchar_t *const paths[1] = { folder.c_str() };
	NVSDK_NGX_FeatureCommonInfo info = {};
	info.PathListInfo.Path = paths;
	info.PathListInfo.Length = 1;
	info.LoggingInfo.LoggingCallback = ngx_log;
	info.LoggingInfo.MinimumLoggingLevel = NVSDK_NGX_LOGGING_LEVEL_ON;
	NVSDK_NGX_Result r = NVSDK_NGX_D3D12_Init_with_ProjectID(k_project, NVSDK_NGX_ENGINE_TYPE_CUSTOM, "1.0", data.c_str(), device, &info);
	if (NVSDK_NGX_FAILED(r))
		return failed(r, "NVIDIA's DLSS runtime could not start (0x%08X)", static_cast<unsigned>(r));
	NVSDK_NGX_Parameter *params = nullptr;
	r = NVSDK_NGX_D3D12_GetCapabilityParameters(&params);
	int available = 0;
	if (NVSDK_NGX_SUCCEED(r) && params != nullptr)
		NVSDK_NGX_Parameter_GetI(params, NVSDK_NGX_Parameter_SuperSamplingDenoising_Available, &available);
	if (!available)
		return failed(r, "Ray Reconstruction is not available on this GPU / driver (0x%08X)", static_cast<unsigned>(r));
	InterlockedExchange(&s_shared->helper_state, rr_state_ngx);
	status("NGX up, waiting for the game's textures");

	NVSDK_NGX_Handle *feature = nullptr;
	LONG opened = 0;
	uint32_t width = 0, height = 0;
	ID3D12Resource *textures[TLOU_RR_SLOTS][rr_kinds] = {};
	ID3D12Resource *constants = nullptr;                    // v78: the slots' frame constants (the add-on's shared buffer)
	int64_t slot_value[TLOU_RR_SLOTS] = {};
	int64_t last = 0;
	uint32_t logged = 0;                                    // v78: the first frames' constants go to the log
	ULONGLONG last_status = GetTickCount64();
	const HANDLE waits[2] = { frame_event, game };
	bool game_ended = false;
	while (s_shared->quit == 0 && !game_ended)
	{
		// v78: a frame published and not taken yet goes first; else wait for the add-on
		if (feature == nullptr || InterlockedCompareExchange64(&s_shared->published, 0, 0) <= last)
		{
			const DWORD woke = WaitForMultipleObjects(2, waits, FALSE, 500);
			if (woke == WAIT_OBJECT_0 + 1 || woke == WAIT_FAILED)
				break;                                          // the game ended
		}
		// a new generation of textures (the render size): RR's feature again
		const LONG generation = s_shared->generation;
		if (generation != 0 && generation != opened)
		{
			gpu_idle(queue, own_fence, own_value, wait_event, 5000);
			if (feature != nullptr)
				NVSDK_NGX_D3D12_ReleaseFeature(feature);
			feature = nullptr;
			for (auto &slot : textures)
				for (ID3D12Resource *&t : slot)
					if (t != nullptr)
					{
						t->Release();
						t = nullptr;
					}
			if (constants != nullptr)
				constants->Release();
			constants = nullptr;
			MemoryBarrier();
			width = s_shared->width;
			height = s_shared->height;
			bool ok = true;
			for (int slot = 0; slot < TLOU_RR_SLOTS && ok; ++slot)
				for (int kind = 0; kind < rr_kinds && ok; ++kind)
				{
					rr_texture_name(name, 160, pid, generation, slot, kind);
					HANDLE handle = nullptr;
					ok = SUCCEEDED(device->OpenSharedHandleByName(name, GENERIC_ALL, &handle)) && SUCCEEDED(device->OpenSharedHandle(handle, IID_PPV_ARGS(&textures[slot][kind])));
					if (handle != nullptr)
						CloseHandle(handle);
				}
			if (ok)
			{
				rr_constants_name(name, 160, pid, generation);
				HANDLE handle = nullptr;
				ok = SUCCEEDED(device->OpenSharedHandleByName(name, GENERIC_ALL, &handle)) && SUCCEEDED(device->OpenSharedHandle(handle, IID_PPV_ARGS(&constants)));
				if (handle != nullptr)
					CloseHandle(handle);
			}
			opened = generation;
			if (!ok)
			{
				status("the game's textures (generation %ld) could not be opened", static_cast<long>(generation));
				continue;
			}
			NVSDK_NGX_DLSSD_Create_Params create = {};
			create.InDenoiseMode = NVSDK_NGX_DLSS_Denoise_Mode_DLUnified;
			create.InRoughnessMode = NVSDK_NGX_DLSS_Roughness_Mode_Packed;
			create.InUseHWDepth = NVSDK_NGX_DLSS_Depth_Type_HW;
			create.InWidth = create.InTargetWidth = width;
			create.InHeight = create.InTargetHeight = height;
			create.InPerfQualityValue = NVSDK_NGX_PerfQuality_Value_DLAA;
			// v78: MVJittered - the add-on's motion vectors run from this frame's jittered pixel to last frame's jittered position
			create.InFeatureCreateFlags = NVSDK_NGX_DLSS_Feature_Flags_IsHDR | NVSDK_NGX_DLSS_Feature_Flags_MVLowRes | NVSDK_NGX_DLSS_Feature_Flags_MVJittered |
				NVSDK_NGX_DLSS_Feature_Flags_AutoExposure;
			allocators[0]->Reset();
			cl->Reset(allocators[0], nullptr);
			r = NGX_D3D12_CREATE_DLSSD_EXT(cl, 1, 1, &feature, params, &create);
			cl->Close();
			ID3D12CommandList *const lists[] = { cl };
			queue->ExecuteCommandLists(1, lists);
			if (NVSDK_NGX_FAILED(r) || feature == nullptr || !gpu_idle(queue, own_fence, own_value, wait_event, 10000))
			{
				feature = nullptr;
				InterlockedExchange(&s_shared->helper_result, static_cast<LONG>(r));
				status("creating Ray Reconstruction at %ux%u failed (0x%08X)", width, height, static_cast<unsigned>(r));
				continue;
			}
			std::fill(std::begin(slot_value), std::end(slot_value), 0);
			last = InterlockedCompareExchange64(&s_shared->published, 0, 0);   // frames of an older generation are not taken
			logged = 0;
			InterlockedExchange(&s_shared->helper_generation, generation);
			InterlockedExchange(&s_shared->helper_state, rr_state_ready);
			status("Ray Reconstruction ready at %ux%u", width, height);
		}
		if (feature == nullptr)
			continue;
		// v78: every published frame in order (the game's queue waits for each output)
		const int64_t newest = InterlockedCompareExchange64(&s_shared->published, 0, 0);
		if (newest <= last)
			continue;
		const int64_t value = last + 1;
		const int slot = static_cast<int>(value % TLOU_RR_SLOTS);
		const rr_frame frame = const_cast<const rr_frame &>(s_shared->frames[slot]);
		const bool recorded = frame.value == value;             // (a value the add-on dropped has no record: no RR, its fence only)
		// the slot's allocator: its last evaluation done
		if (slot_value[slot] != 0 && fence_out->GetCompletedValue() < static_cast<UINT64>(slot_value[slot]))
		{
			fence_out->SetEventOnCompletion(static_cast<UINT64>(slot_value[slot]), wait_event);
			if (WaitForSingleObject(wait_event, 2000) != WAIT_OBJECT_0)
			{
				status("the GPU is behind (frame %lld)", static_cast<long long>(value));
				continue;
			}
		}
		allocators[slot]->Reset();
		// v78: the frame's constants - the queue waits for the game's inputs, copies the slot's constants to the readback
		// buffer, and this CPU waits for them (the game's frame may take its time: a loading screen, the game in the background)
		copy_list->Reset(allocators[slot], nullptr);
		copy_list->CopyBufferRegion(readback, UINT64(slot) * TLOU_RR_CONSTANT_BYTES, constants, UINT64(slot) * TLOU_RR_CONSTANT_BYTES, TLOU_RR_CONSTANT_BYTES);
		copy_list->Close();
		queue->Wait(fence_in, static_cast<UINT64>(value));      // the game's inputs for this frame
		{
			ID3D12CommandList *const lists[] = { copy_list };
			queue->ExecuteCommandLists(1, lists);
		}
		++own_value;
		queue->Signal(own_fence, own_value);
		own_fence->SetEventOnCompletion(own_value, wait_event);
		bool arrived = false;
		while (!arrived && s_shared->quit == 0)
		{
			const HANDLE inputs[2] = { wait_event, game };
			const DWORD woke = WaitForMultipleObjects(2, inputs, FALSE, 500);
			if (woke == WAIT_OBJECT_0)
				arrived = true;
			else if (woke == WAIT_OBJECT_0 + 1 || woke == WAIT_FAILED)
			{
				game_ended = true;
				break;
			}
		}
		if (!arrived)
			break;
		rr_gpu_frame g = {};
		std::memcpy(&g, readback_ptr + UINT64(slot) * TLOU_RR_CONSTANT_BYTES, sizeof(g));
		if (logged < 32 && recorded)
		{
			++logged;
			say("frame %lld (add-on frame %u): jitter %+.4f %+.4f px, reset %u%s", static_cast<long long>(value), g.frame, g.jitter_x, g.jitter_y,
				g.reset | static_cast<uint32_t>(frame.reset), logged == 32 ? " (no more frames logged)" : "");
		}
		r = NVSDK_NGX_Result_Success;
		if (recorded)
		{
			cl->Reset(allocators[slot], nullptr);
			ID3D12Resource *const *const t = textures[slot];
			for (int kind = 0; kind < rr_kinds; ++kind)
				barrier(cl, t[kind], D3D12_RESOURCE_STATE_COMMON, kind == rr_output ? D3D12_RESOURCE_STATE_UNORDERED_ACCESS :
					D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE | D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
			// v78: the matrices once the add-on gives them (all 0: not given yet - identities, as v61-v77 passed)
			float world_to_view[16] = {}, view_to_clip[16] = {};
			bool given = false;
			for (int i = 0; i < 16; ++i)
				given = given || g.world_to_view[i] != 0.0f;
			for (int i = 0; i < 16; ++i)
			{
				world_to_view[i] = given ? g.world_to_view[i] : (i % 5 == 0 ? 1.0f : 0.0f);
				view_to_clip[i] = given ? g.view_to_clip[i] : (i % 5 == 0 ? 1.0f : 0.0f);
			}
			NVSDK_NGX_D3D12_DLSSD_Eval_Params eval = {};
			eval.pInColor = t[rr_color];
			eval.pInOutput = t[rr_output];
			eval.pInDepth = t[rr_depth];
			eval.pInMotionVectors = t[rr_motion];
			eval.pInDiffuseAlbedo = t[rr_diffuse_albedo];
			eval.pInSpecularAlbedo = t[rr_specular_albedo];
			eval.pInNormals = t[rr_normals];
			eval.pInSpecularHitDistance = nullptr;              // v78: not yet (with the matrices, v79)
			eval.InJitterOffsetX = g.jitter_x;
			eval.InJitterOffsetY = g.jitter_y;
			eval.InRenderSubrectDimensions = { width, height };
			eval.InReset = (g.reset != 0 || frame.reset != 0) ? 1 : 0;
			eval.InMVScaleX = g.mv_scale_x != 0.0f ? g.mv_scale_x : 1.0f;
			eval.InMVScaleY = g.mv_scale_y != 0.0f ? g.mv_scale_y : 1.0f;
			eval.InPreExposure = g.pre_exposure > 0.0f ? g.pre_exposure : 1.0f;
			eval.pInWorldToViewMatrix = world_to_view;
			eval.pInViewToClipMatrix = view_to_clip;
			r = NGX_D3D12_EVALUATE_DLSSD_EXT(cl, feature, params, &eval);
			for (int kind = 0; kind < rr_kinds; ++kind)
				barrier(cl, t[kind], kind == rr_output ? D3D12_RESOURCE_STATE_UNORDERED_ACCESS :
					D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE | D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_COMMON);
			cl->Close();
			if (NVSDK_NGX_SUCCEED(r))
			{
				ID3D12CommandList *const lists[] = { cl };
				queue->ExecuteCommandLists(1, lists);
			}
		}
		queue->Signal(fence_out, static_cast<UINT64>(value));   // (also on a failed evaluation: the game's queue waits for it)
		slot_value[slot] = value;
		last = value;
		InterlockedExchange64(&s_shared->submitted, value);
		if (recorded && NVSDK_NGX_SUCCEED(r))
			InterlockedIncrement64(&s_shared->evaluated);
		InterlockedExchange(&s_shared->helper_result, static_cast<LONG>(r));
		if (GetTickCount64() - last_status > 2000)
		{
			last_status = GetTickCount64();
			status("evaluating at %ux%u: %lld frames, jitter %+.3f %+.3f%s", width, height, static_cast<long long>(s_shared->evaluated), g.jitter_x, g.jitter_y,
				NVSDK_NGX_FAILED(r) ? " (the last evaluation failed)" : "");
		}
	}
	say("exiting (game %s, quit %ld)", WaitForSingleObject(game, 0) == WAIT_OBJECT_0 ? "ended" : "running", static_cast<long>(s_shared->quit));
	gpu_idle(queue, own_fence, own_value, wait_event, 2000);
	if (feature != nullptr)
		NVSDK_NGX_D3D12_ReleaseFeature(feature);
	NVSDK_NGX_D3D12_DestroyParameters(params);
	NVSDK_NGX_D3D12_Shutdown1(device);                      // this process's own NGX (not the game's)
	return 0;
}
