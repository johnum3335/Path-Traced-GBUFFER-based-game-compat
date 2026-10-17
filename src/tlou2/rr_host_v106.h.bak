// TLOU RT - Ray Reconstruction helper, the add-on's side (v61; protocol: rr_shared.h). Creates the shared memory, the frame
// event, the two shared fences, the shared textures and (v78) the shared constants buffer on the game's device, starts
// tlou-rt-rr.exe in a job that ends it with the game, and publishes frames. v78: the game's queue waits on the output fence
// before a lighting list that takes RR's output (queue_wait); a watcher thread ends such a wait from the CPU when the helper
// is gone or k_wait_limit_ms late - the game never hangs on the helper - and times every frame from its publishing to its
// output (RR's latency). Apart from the watcher not thread-safe: one owner at a time (the add-on's s_rr_mutex).
#pragma once
#include "rr_shared.h"
#include <d3d12.h>
#include <algorithm>
#include <atomic>
#include <cstring>
#include <deque>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

struct rr_host
{
	static constexpr double k_wait_limit_ms = 1000.0;      // v78: a game-queue wait the helper has not ended by then is ended here

	DWORD pid = GetCurrentProcessId();
	HANDLE mapping = nullptr, frame_event = nullptr, job = nullptr, process = nullptr;
	rr_shared *shared = nullptr;
	ID3D12Fence *fence_in = nullptr, *fence_out = nullptr;
	HANDLE fence_in_handle = nullptr, fence_out_handle = nullptr;
	ID3D12Resource *textures[TLOU_RR_SLOTS][rr_kinds] = {};
	HANDLE texture_handles[TLOU_RR_SLOTS][rr_kinds] = {};
	ID3D12Resource *constants = nullptr;                   // v78: the slots' rr_gpu_frame (TLOU_RR_SLOTS x TLOU_RR_CONSTANT_BYTES)
	HANDLE constants_handle = nullptr;
	uint32_t width = 0, height = 0;
	LONG generation = 0;
	int64_t published = 0;
	std::string error;

	// v78: the watcher's state (watch_mutex): the frames published and not done yet (value, QPC when published), the highest
	// value the game's queue waits for and when that wait was queued, the latencies since take_stats (ms), the counts
	struct stats
	{
		size_t frames = 0;
		float median = 0.0f, p90 = 0.0f;
		uint64_t waits = 0, released = 0;
	};
	std::mutex watch_mutex;
	std::deque<std::pair<int64_t, int64_t>> watch_published;
	int64_t watch_wait = 0, watch_wait_qpc = 0;
	std::vector<float> watch_latency;
	uint64_t watch_waits = 0, watch_released = 0;
	HANDLE watch_event = nullptr, watch_kick = nullptr;
	std::thread watcher;
	std::atomic<bool> watch_quit { false };

	// v78: a host still running at the end (the process exiting without the device's destruction: its threads are gone) lets
	// the watcher go - a joinable std::thread's destructor would end the process with std::terminate
	~rr_host()
	{
		if (watcher.joinable())
		{
			watch_quit = true;
			watcher.detach();
		}
	}

	bool started() const { return shared != nullptr; }

	bool helper_alive() const
	{
		return process != nullptr && WaitForSingleObject(process, 0) == WAIT_TIMEOUT;
	}

	DWORD helper_exit_code() const
	{
		DWORD code = 0;
		if (process != nullptr)
			GetExitCodeProcess(process, &code);
		return code;
	}

	int64_t completed() const
	{
		if (fence_out == nullptr)
			return 0;
		const UINT64 value = fence_out->GetCompletedValue();
		return value == UINT64_MAX ? INT64_MAX : static_cast<int64_t>(value);   // (UINT64_MAX: the device was removed)
	}

	// The helper is done with everything published (or gone): the textures may be replaced
	bool drained() const
	{
		return !helper_alive() || completed() >= published;
	}

	// RR's feature exists for the current textures: frames can be published
	bool ready() const
	{
		return shared != nullptr && generation != 0 && helper_alive() && shared->helper_state == rr_state_ready && shared->helper_generation == generation;
	}

	// v78: frame value's slot may take its inputs - the helper finished the frame that used it last
	bool slot_free(int64_t value) const
	{
		return completed() >= value - TLOU_RR_SLOTS;
	}

	bool start(ID3D12Device *device, const std::wstring &helper_exe, const std::wstring &helper_dir)
	{
		wchar_t name[160];
		rr_name(name, 160, pid, L"shared");
		mapping = CreateFileMappingW(INVALID_HANDLE_VALUE, nullptr, PAGE_READWRITE, 0, sizeof(rr_shared), name);
		shared = mapping != nullptr ? static_cast<rr_shared *>(MapViewOfFile(mapping, FILE_MAP_ALL_ACCESS, 0, 0, sizeof(rr_shared))) : nullptr;
		if (shared == nullptr)
			return fail("the shared memory could not be created");
		ZeroMemory(const_cast<rr_shared *>(shared), sizeof(rr_shared));
		shared->magic = TLOU_RR_MAGIC;
		shared->version = TLOU_RR_VERSION;
		shared->adapter = device->GetAdapterLuid();
		rr_name(name, 160, pid, L"frame");
		frame_event = CreateEventW(nullptr, FALSE, FALSE, name);
		if (frame_event == nullptr)
			return fail("the frame event could not be created");
		if (!shared_fence(device, L"fence-in", fence_in, fence_in_handle) || !shared_fence(device, L"fence-out", fence_out, fence_out_handle))
			return fail("the shared fences could not be created");
		// the helper, in a job that ends it with this process
		job = CreateJobObjectW(nullptr, nullptr);
		if (job != nullptr)
		{
			JOBOBJECT_EXTENDED_LIMIT_INFORMATION limits = {};
			limits.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
			SetInformationJobObject(job, JobObjectExtendedLimitInformation, &limits, sizeof(limits));
		}
		std::wstring command = L"\"" + helper_exe + L"\" " + std::to_wstring(pid);
		STARTUPINFOW si = { sizeof(si) };
		PROCESS_INFORMATION pi = {};
		if (!CreateProcessW(helper_exe.c_str(), command.data(), nullptr, nullptr, FALSE, CREATE_SUSPENDED | CREATE_NO_WINDOW, nullptr, helper_dir.c_str(), &si, &pi))
			return fail("the helper could not be started (error " + std::to_string(GetLastError()) + ")");
		if (job != nullptr)
			AssignProcessToJobObject(job, pi.hProcess);
		ResumeThread(pi.hThread);
		CloseHandle(pi.hThread);
		process = pi.hProcess;
		// v78: the watcher
		watch_event = CreateEventW(nullptr, FALSE, FALSE, nullptr);
		watch_kick = CreateEventW(nullptr, FALSE, FALSE, nullptr);
		if (watch_event == nullptr || watch_kick == nullptr)
			return fail("the watcher's events could not be created");
		watch_quit = false;
		watcher = std::thread([this]() { watch(); });
		return true;
	}

	// A new generation of shared textures (and v78 constants) at w x h (only when drained()); the old ones go to retired
	// (released by the caller once the game's GPU work that used them is done)
	bool resize(ID3D12Device *device, uint32_t w, uint32_t h, std::vector<IUnknown *> &retired)
	{
		release_textures(retired);
		const LONG next = generation + 1;
		wchar_t name[160];
		for (int slot = 0; slot < TLOU_RR_SLOTS; ++slot)
			for (int kind = 0; kind < rr_kinds; ++kind)
			{
				D3D12_HEAP_PROPERTIES heap = { D3D12_HEAP_TYPE_DEFAULT };
				D3D12_RESOURCE_DESC desc = {};
				desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
				desc.Width = w;
				desc.Height = h;
				desc.DepthOrArraySize = 1;
				desc.MipLevels = 1;
				desc.Format = k_rr_formats[kind];
				desc.SampleDesc.Count = 1;
				desc.Flags = D3D12_RESOURCE_FLAG_ALLOW_SIMULTANEOUS_ACCESS | (kind == rr_output ? D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS : D3D12_RESOURCE_FLAG_NONE);
				rr_texture_name(name, 160, pid, next, slot, kind);
				if (FAILED(device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_SHARED, &desc, D3D12_RESOURCE_STATE_COMMON, nullptr, IID_PPV_ARGS(&textures[slot][kind]))) ||
					FAILED(device->CreateSharedHandle(textures[slot][kind], nullptr, GENERIC_ALL, name, &texture_handles[slot][kind])))
				{
					release_textures(retired);
					return fail("the shared textures could not be created");
				}
			}
		{
			D3D12_HEAP_PROPERTIES heap = { D3D12_HEAP_TYPE_DEFAULT };
			D3D12_RESOURCE_DESC desc = {};
			desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
			desc.Width = UINT64(TLOU_RR_SLOTS) * TLOU_RR_CONSTANT_BYTES;
			desc.Height = 1;
			desc.DepthOrArraySize = 1;
			desc.MipLevels = 1;
			desc.SampleDesc.Count = 1;
			desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
			rr_constants_name(name, 160, pid, next);
			if (FAILED(device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_SHARED, &desc, D3D12_RESOURCE_STATE_COMMON, nullptr, IID_PPV_ARGS(&constants))) ||
				FAILED(device->CreateSharedHandle(constants, nullptr, GENERIC_ALL, name, &constants_handle)))
			{
				release_textures(retired);
				return fail("the shared constants could not be created");
			}
		}
		width = w;
		height = h;
		generation = next;
		shared->width = w;
		shared->height = h;
		MemoryBarrier();
		InterlockedExchange(&shared->generation, next);
		SetEvent(frame_event);
		return true;
	}

	// v78: publishes frame value (above published; a value the add-on dropped goes with it - the helper passes over a value
	// without its record): the game's queue signals the input fence to it after the work submitted so far, which wrote the
	// slot's inputs
	bool publish_value(ID3D12CommandQueue *queue, int64_t value, rr_frame frame)
	{
		if (shared == nullptr || fence_in == nullptr)
			return false;
		if (value <= published)
			return true;
		frame.value = value;
		const_cast<rr_frame &>(shared->frames[value % TLOU_RR_SLOTS]) = frame;
		MemoryBarrier();
		if (FAILED(queue->Signal(fence_in, static_cast<UINT64>(value))))
			return false;
		published = value;
		InterlockedExchange64(&shared->published, value);
		SetEvent(frame_event);
		LARGE_INTEGER now;
		QueryPerformanceCounter(&now);
		{
			const std::lock_guard<std::mutex> lock(watch_mutex);
			watch_published.emplace_back(value, now.QuadPart);
			while (watch_published.size() > 64)
				watch_published.pop_front();
		}
		SetEvent(watch_kick);
		return true;
	}

	// v78: the game's queue waits for frame value's output (queued before the list that reprojects it); the watcher ends the
	// wait from the CPU should the helper not
	bool queue_wait(ID3D12CommandQueue *queue, int64_t value)
	{
		if (fence_out == nullptr || FAILED(queue->Wait(fence_out, static_cast<UINT64>(value))))
			return false;
		LARGE_INTEGER now;
		QueryPerformanceCounter(&now);
		{
			const std::lock_guard<std::mutex> lock(watch_mutex);
			if (value > watch_wait)
			{
				watch_wait = value;
				watch_wait_qpc = now.QuadPart;
			}
			++watch_waits;
		}
		SetEvent(watch_kick);
		return true;
	}

	// v78: RR's latency (publishing to output) of the frames done since the last call, and the waits so far
	stats take_stats()
	{
		stats s;
		std::vector<float> v;
		{
			const std::lock_guard<std::mutex> lock(watch_mutex);
			v.swap(watch_latency);
			s.waits = watch_waits;
			s.released = watch_released;
		}
		s.frames = v.size();
		if (!v.empty())
		{
			std::sort(v.begin(), v.end());
			s.median = v[v.size() / 2];
			s.p90 = v[std::min(v.size() - 1, v.size() * 9 / 10)];
		}
		return s;
	}

	// Ends the helper (its job) and hands every GPU object to retired. v78: the watcher first, then whatever the game's queue
	// could still wait for is ended
	void stop(std::vector<IUnknown *> &retired)
	{
		if (watcher.joinable())
		{
			watch_quit = true;
			SetEvent(watch_kick);
			watcher.join();
		}
		if (fence_out != nullptr)
		{
			const int64_t top = std::max(published, watch_wait);
			if (top > completed())
				fence_out->Signal(static_cast<UINT64>(top));
		}
		if (shared != nullptr)
			InterlockedExchange(&shared->quit, 1);
		if (frame_event != nullptr)
			SetEvent(frame_event);
		if (process != nullptr && WaitForSingleObject(process, 2000) == WAIT_TIMEOUT)
			TerminateProcess(process, 1);
		for (HANDLE *h : { &process, &job, &watch_event, &watch_kick })
			if (*h != nullptr)
			{
				CloseHandle(*h);
				*h = nullptr;
			}
		release_textures(retired);
		for (ID3D12Fence **f : { &fence_in, &fence_out })
			if (*f != nullptr)
			{
				retired.push_back(*f);
				*f = nullptr;
			}
		for (HANDLE *h : { &fence_in_handle, &fence_out_handle, &frame_event })
			if (*h != nullptr)
			{
				CloseHandle(*h);
				*h = nullptr;
			}
		if (shared != nullptr)
			UnmapViewOfFile(const_cast<rr_shared *>(shared));
		if (mapping != nullptr)
			CloseHandle(mapping);
		shared = nullptr;
		mapping = nullptr;
		width = height = 0;
		generation = 0;
		published = 0;
		watch_published.clear();
		watch_wait = watch_wait_qpc = 0;
		watch_latency.clear();
		watch_waits = watch_released = 0;
	}

	std::string helper_message() const
	{
		if (shared == nullptr)
			return std::string();
		char text[sizeof(shared->message) + 1] = {};
		std::memcpy(text, const_cast<const char *>(shared->message), sizeof(shared->message));
		return text;
	}

private:
	bool fail(const std::string &message)
	{
		error = message;
		return false;
	}

	bool shared_fence(ID3D12Device *device, const wchar_t *what, ID3D12Fence *&fence, HANDLE &handle)
	{
		wchar_t name[160];
		rr_name(name, 160, pid, what);
		return SUCCEEDED(device->CreateFence(0, D3D12_FENCE_FLAG_SHARED, IID_PPV_ARGS(&fence))) && SUCCEEDED(device->CreateSharedHandle(fence, nullptr, GENERIC_ALL, name, &handle));
	}

	void release_textures(std::vector<IUnknown *> &retired)
	{
		for (int slot = 0; slot < TLOU_RR_SLOTS; ++slot)
			for (int kind = 0; kind < rr_kinds; ++kind)
			{
				if (textures[slot][kind] != nullptr)
					retired.push_back(textures[slot][kind]);
				textures[slot][kind] = nullptr;
				if (texture_handles[slot][kind] != nullptr)
					CloseHandle(texture_handles[slot][kind]);
				texture_handles[slot][kind] = nullptr;
			}
		if (constants != nullptr)
			retired.push_back(constants);
		constants = nullptr;
		if (constants_handle != nullptr)
			CloseHandle(constants_handle);
		constants_handle = nullptr;
	}

	// v78: the watcher thread - waits for the oldest frame published (or the game queue's wait) on the output fence, times
	// each frame from its publishing to its output, and ends the game queue's wait from the CPU when the helper is gone or
	// k_wait_limit_ms late (its output for that frame may then be incomplete: one frame's light)
	void watch()
	{
		LARGE_INTEGER frequency;
		QueryPerformanceFrequency(&frequency);
		const double to_ms = 1000.0 / static_cast<double>(frequency.QuadPart);
		int64_t registered = 0;
		while (!watch_quit.load())
		{
			int64_t target = 0;
			{
				const std::lock_guard<std::mutex> lock(watch_mutex);
				target = !watch_published.empty() ? watch_published.front().first : watch_wait;
			}
			if (target != 0 && target != registered)
			{
				fence_out->SetEventOnCompletion(static_cast<UINT64>(target), watch_event);
				registered = target;
			}
			if (target != 0)
			{
				const HANDLE handles[2] = { watch_event, watch_kick };
				WaitForMultipleObjects(2, handles, FALSE, 10);
			}
			else
				WaitForSingleObject(watch_kick, 100);
			LARGE_INTEGER now;
			QueryPerformanceCounter(&now);
			const int64_t done = completed();
			const bool alive = helper_alive();
			const std::lock_guard<std::mutex> lock(watch_mutex);
			while (!watch_published.empty() && (watch_published.front().first <= done || !alive))
			{
				if (alive)
					watch_latency.push_back(static_cast<float>((now.QuadPart - watch_published.front().second) * to_ms));
				watch_published.pop_front();
			}
			if (watch_latency.size() > 4096)
				watch_latency.erase(watch_latency.begin(), watch_latency.begin() + 2048);
			if (watch_wait != 0 && done >= watch_wait)
				watch_wait = 0;
			else if (watch_wait != 0 && (!alive || (now.QuadPart - watch_wait_qpc) * to_ms > k_wait_limit_ms))
			{
				fence_out->Signal(static_cast<UINT64>(watch_wait));   // the game's queue goes on
				++watch_released;
				watch_wait = 0;
			}
		}
	}
};
