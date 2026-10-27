#pragma once

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <tuple>
#include <vector>
#include <unistd.h>

#include <hggc.h>

#include <ATen/ATen.h>
#include <ATen/DeviceAccelerator.h>
#include <c10/core/DeviceGuard.h>
#include <c10/util/ScopeExit.h>

#define MOONEP_HGCHECK(cmd) do {                                \
    HGresult err = cmd;                                        \
    if (err != HGGC_SUCCESS) {                                 \
        const char *errStr = "unknown driver error";            \
        hgGetErrorString(err, &errStr);                         \
        TORCH_CHECK(false, #cmd, " failed (", int(err), "): ",  \
                    errStr ? errStr : "unknown driver error"); \
    }                                                         \
} while(0)

static inline c10::DeviceType ptg_device_type() {
    return at::accelerator::getAccelerator(true).value();
}

static inline int ptg_current_device() {
    return at::accelerator::getDeviceIndex();
}

static inline at::TensorOptions ptg_tensor_options(
    at::ScalarType dtype, int device_id
) {
    return at::TensorOptions()
        .dtype(dtype)
        .device(c10::Device(ptg_device_type(), device_id));
}

static inline void ptg_synchronize(int device_id) {
    at::accelerator::synchronizeDevice(device_id);
}

static inline at::Tensor ptg_make_vmm_tensor(
    void *ptr,
    size_t nbytes,
    const std::vector<int64_t> &shape,
    at::ScalarType dtype,
    int device_id,
    std::function<void(void *)> deleter
) {
    auto options = ptg_tensor_options(dtype, device_id);
    auto storage = c10::Storage(
        c10::Storage::use_byte_size_t(),
        static_cast<int64_t>(nbytes),
        at::InefficientStdFunctionContext::makeDataPtr(
            ptr, std::move(deleter), options.device()),
        /*allocator=*/nullptr,
        /*resizable=*/false);

    auto impl = c10::make_intrusive<at::TensorImpl>(
        std::move(storage),
        c10::DispatchKeySet(options.computeDispatchKey()),
        c10::scalarTypeToTypeMeta(dtype));
    impl->set_sizes_contiguous(shape);
    return at::Tensor(std::move(impl));
}

static inline bool icn_fabric_supported_for(int device_id) {
    static std::mutex mutex;
    static std::map<int, bool> cache;
    std::lock_guard<std::mutex> lock(mutex);
    const auto cached = cache.find(device_id);
    if (cached != cache.end()) return cached->second;

    HGdevice hg_device;
    MOONEP_HGCHECK(hgDeviceGet(&hg_device, device_id));
    int supported = 0;
    HGresult err = hgDeviceGetAttribute(&supported,
        HG_DEVICE_ATTRIBUTE_HANDLE_TYPE_FABRIC_SUPPORTED, hg_device);
    // Query/probe failures may be transient; only cache a definite result.
    if (err != HGGC_SUCCESS) return false;
    if (!supported) {
        cache.emplace(device_id, false);
        return false;
    }

    HGmemAllocationProp prop = {};
    prop.type = HG_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type = HG_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id = device_id;
    prop.requestedHandleTypes = HG_MEM_HANDLE_TYPE_FABRIC;
    size_t size;
    if (hgMemGetAllocationGranularity(
            &size, &prop, HG_MEM_ALLOC_GRANULARITY_RECOMMENDED) != HGGC_SUCCESS)
        return false;

    HGmemGenericAllocationHandle handle;
    if (hgMemCreate(&handle, size, &prop, 0) != HGGC_SUCCESS) return false;
    MOONEP_HGCHECK(hgMemRelease(handle));
    cache.emplace(device_id, true);
    return true;
}

static inline bool icn_fabric_supported() {
    return icn_fabric_supported_for(ptg_current_device());
}

static inline size_t icn_granularity_for(int device_id,
                                         HGmemAllocationHandleType ht) {
    static std::mutex mutex;
    static std::map<std::pair<int, HGmemAllocationHandleType>, size_t> cache;
    std::lock_guard<std::mutex> lock(mutex);
    const auto key = std::make_pair(device_id, ht);
    const auto cached = cache.find(key);
    if (cached != cache.end()) return cached->second;

    HGmemAllocationProp prop = {};
    prop.type = HG_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type = HG_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id = device_id;
    prop.requestedHandleTypes = ht;

    size_t granularity;
    MOONEP_HGCHECK(hgMemGetAllocationGranularity(
        &granularity, &prop, HG_MEM_ALLOC_GRANULARITY_RECOMMENDED));
    cache.emplace(key, granularity);
    return granularity;
}

static inline size_t icn_granularity_max(int device_id) {
    size_t gran = icn_granularity_for(
        device_id, HG_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR);
    if (icn_fabric_supported_for(device_id)) {
        gran = std::max(
            gran, icn_granularity_for(device_id, HG_MEM_HANDLE_TYPE_FABRIC));
    }
    return gran;
}

static inline HGmemAllocationHandleType icn_handle_type(bool use_fabric) {
    return use_fabric ? HG_MEM_HANDLE_TYPE_FABRIC
                      : HG_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
}

static inline std::tuple<size_t, size_t, int> icn_prepare(
    const std::vector<int64_t> &shape,
    at::ScalarType dtype,
    bool use_fabric
) {
    TORCH_CHECK(!shape.empty(), "Shape must be non-empty");

    const int device_id = ptg_current_device();

    size_t size = c10::elementSize(dtype);
    for (auto dim : shape) {
        TORCH_CHECK(dim > 0, "Size dimensions must be positive");
        size *= static_cast<size_t>(dim);
    }

    size_t granularity = icn_granularity_for(
        device_id, icn_handle_type(use_fabric));
    size_t allocated_size = (size + granularity - 1) / granularity * granularity;

    return {size, allocated_size, device_id};
}

static inline void icn_cleanup_check(HGresult result, const char *operation) noexcept {
    if (result != HGGC_SUCCESS)
        std::fprintf(stderr, "MoonEP VMM cleanup: %s failed (%d)\n", operation, int(result));
}

// Shared ownership also covers exceptions during Tensor/Storage construction.
struct IcnMapping {
    int device_id;
    size_t chunk_size;
    size_t total_size;
    HGdeviceptr ptr = 0;
    size_t mapped_chunks = 0;
    bool reserved = false;

    IcnMapping(int device, size_t chunk, size_t total)
        : device_id(device), chunk_size(chunk), total_size(total) {}

    ~IcnMapping() noexcept {
        if (!reserved) return;
        try {
            c10::DeviceGuard guard(c10::Device(ptg_device_type(), device_id));
            try {
                if (mapped_chunks) ptg_synchronize(device_id);
            } catch (const std::exception &error) {
                std::fprintf(stderr, "MoonEP VMM cleanup synchronization: %s\n", error.what());
            }
            for (size_t i = 0; i < mapped_chunks; ++i)
                icn_cleanup_check(hgMemUnmap(ptr + i * chunk_size, chunk_size), "hgMemUnmap");
            icn_cleanup_check(hgMemAddressFree(ptr, total_size), "hgMemAddressFree");
        } catch (const std::exception &error) {
            std::fprintf(stderr, "MoonEP VMM cleanup: %s\n", error.what());
        }
    }

    void reserve() {
        MOONEP_HGCHECK(hgMemAddressReserve(&ptr, total_size, 0, 0, 0));
        reserved = true;
    }

    void map(HGmemGenericAllocationHandle handle) {
        HGdeviceptr chunk = ptr + mapped_chunks * chunk_size;
        MOONEP_HGCHECK(hgMemMap(chunk, chunk_size, 0, handle, 0));
        ++mapped_chunks;
        HGmemAccessDesc access = {};
        access.location.type = HG_MEM_LOCATION_TYPE_DEVICE;
        access.location.id = device_id;
        access.flags = HG_MEM_ACCESS_FLAGS_PROT_READWRITE;
        MOONEP_HGCHECK(hgMemSetAccess(chunk, chunk_size, &access, 1));
    }
};

static constexpr int64_t kFabricHandleBytes =
    static_cast<int64_t>(sizeof(HGmemFabricHandle::data));

// Export an allocation as a CPU tensor matching
// `use_fabric`: uint8[64] holding a HGmemFabricHandle, or int64[] (a scalar
// tensor) holding a POSIX fd.
static inline at::Tensor icn_export_shareable(
    HGmemGenericAllocationHandle handle, bool use_fabric
) {
    if (use_fabric) {
        HGmemFabricHandle fabric = {};
        MOONEP_HGCHECK(hgMemExportToShareableHandle(
            &fabric, handle, HG_MEM_HANDLE_TYPE_FABRIC, 0));
        auto out = at::empty({kFabricHandleBytes},
                             at::TensorOptions().dtype(at::kByte));
        std::memcpy(out.data_ptr(), fabric.data, sizeof(fabric.data));
        return out;
    }
    int fd = -1;
    MOONEP_HGCHECK(hgMemExportToShareableHandle(
        &fd, handle, HG_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR, 0));
    TORCH_CHECK(fd >= 0, "hgMemExportToShareableHandle returned invalid fd");
    auto close_fd = c10::make_scope_exit([fd] { ::close(fd); });
    auto tensor = at::scalar_tensor(fd, at::TensorOptions().dtype(at::kLong));
    close_fd.release();
    return tensor;
}

static inline void icn_check_handles(
    const at::Tensor &handles, int64_t world_size, bool use_fabric
) {
    TORCH_CHECK(handles.device().is_cpu() && handles.is_contiguous(),
        "shareable handles must be a contiguous CPU tensor, got device=",
        handles.device(), " contiguous=", handles.is_contiguous());
    if (use_fabric) {
        TORCH_CHECK(handles.scalar_type() == at::kByte && handles.dim() == 2
                    && handles.size(0) == world_size
                    && handles.size(1) == kFabricHandleBytes,
            "fabric handles must be uint8[", world_size, ", ",
            kFabricHandleBytes, "], got ", handles.scalar_type(), handles.sizes());
    } else {
        TORCH_CHECK(handles.scalar_type() == at::kLong
                    && handles.numel() == world_size,
            "fd handles must be int64 with ", world_size, " elements, got ",
            handles.scalar_type(), handles.sizes());
    }
}

// Import row `index` of a handle tensor already validated by icn_check_handles.
static inline HGmemGenericAllocationHandle icn_import_shareable(
    const at::Tensor &handles, int64_t index, bool use_fabric
) {
    HGmemGenericAllocationHandle handle;
    if (use_fabric) {
        HGmemFabricHandle fabric = {};
        std::memcpy(fabric.data,
                    handles.const_data_ptr<uint8_t>() + index * kFabricHandleBytes,
                    sizeof(fabric.data));
        MOONEP_HGCHECK(hgMemImportFromShareableHandle(
            &handle, &fabric, HG_MEM_HANDLE_TYPE_FABRIC));
    } else {
        int64_t fd = handles.const_data_ptr<int64_t>()[index];
        TORCH_CHECK(fd >= 0, "invalid fd ", fd, " at index ", index);
        MOONEP_HGCHECK(hgMemImportFromShareableHandle(
            &handle, reinterpret_cast<void *>(static_cast<intptr_t>(fd)),
            HG_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR));
    }
    return handle;
}

// Returns (keepalive VA tensor, shareable handle, owned mem handle as int64).
// The caller must release the owned handle after all required mappings exist;
// the physical allocation then remains alive through those mappings.
// The shareable handle is what peers import: an int64 scalar tensor holding an
// fd the caller must close once all peers have imported it, or a uint8[64]
// fabric handle that needs no cleanup.
inline std::tuple<at::Tensor, at::Tensor, int64_t> icn_dist_alloc(
    const std::vector<int64_t> &chunk_shape,
    at::ScalarType dtype,
    bool use_fabric
) {
    auto [nbytes, allocated_size, device_id] =
        icn_prepare(chunk_shape, dtype, use_fabric);

    HGmemAllocationProp prop = {};
    prop.type = HG_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type = HG_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id = device_id;
    prop.requestedHandleTypes = icn_handle_type(use_fabric);

    HGmemGenericAllocationHandle mem_handle;
    MOONEP_HGCHECK(hgMemCreate(&mem_handle, allocated_size, &prop, 0));
    auto release_handle = c10::make_scope_exit([mem_handle] {
        icn_cleanup_check(hgMemRelease(mem_handle), "hgMemRelease");
    });
    auto mapping = std::make_shared<IcnMapping>(device_id, allocated_size, allocated_size);
    mapping->reserve();
    mapping->map(mem_handle);

    // The captured shared_ptr releases the mapping when the Storage is destroyed.
    auto keepalive = ptg_make_vmm_tensor(
        reinterpret_cast<void *>(mapping->ptr), nbytes, chunk_shape, dtype,
        device_id, [mapping](void *) {});

    // Export for other processes to import. The caller releases mem_handle
    // after every required mapping has been established.
    auto shareable = icn_export_shareable(mem_handle, use_fabric);

    auto close_fd = c10::make_scope_exit([&] {
        if (!use_fabric) ::close(static_cast<int>(shareable.item<int64_t>()));
    });
    auto result = std::make_tuple(keepalive, shareable, static_cast<int64_t>(mem_handle));
    close_fd.release();
    release_handle.release();
    return result;
}

// Release the owned mem handle returned by icn_dist_alloc (the physical memory
// stays alive via the existing maps).
static inline void icn_release_mem_handle(int64_t mem_handle_u64) {
    MOONEP_HGCHECK(hgMemRelease(static_cast<HGmemGenericAllocationHandle>(mem_handle_u64)));
}

/**
 * Map all chunk handles into a contiguous VA region.
 * All chunks get RW access because dispatch writes to remote ranks' regions.
 *
 * `shareables` are the handles exported by each rank via icn_dist_alloc and
 * routed to this process by the caller: POSIX fds (already valid here, the
 * caller may close them once imported) or 64-byte fabric blobs.
 */
inline at::Tensor icn_dist_map(
    const std::vector<int64_t> &chunk_shape,
    at::ScalarType dtype,
    const at::Tensor &shareables,
    int64_t local_rank,
    int64_t world_size,
    bool use_fabric
) {
    TORCH_CHECK(world_size > 0 && local_rank >= 0 && local_rank < world_size,
        "Invalid local rank or world size");
    icn_check_handles(shareables, world_size, use_fabric);

    auto [chunk_nbytes, chunk_allocated_size, device_id] =
        icn_prepare(chunk_shape, dtype, use_fabric);

    TORCH_CHECK(chunk_allocated_size == chunk_nbytes,
        "Chunk byte size (", chunk_nbytes, ") must be aligned to VMM granularity (",
        chunk_allocated_size, "). Adjust chunk dimensions so that the total bytes "
        "are a multiple of the allocation granularity.");

    size_t total_allocated_size = chunk_allocated_size * world_size;

    auto mapping = std::make_shared<IcnMapping>(
        device_id, chunk_allocated_size, total_allocated_size);
    mapping->reserve();

    for (int64_t i = 0; i < world_size; i++) {
        HGmemGenericAllocationHandle mem_handle =
            icn_import_shareable(shareables, i, use_fabric);

        auto release_handle = c10::make_scope_exit([mem_handle] {
            icn_cleanup_check(hgMemRelease(mem_handle), "hgMemRelease");
        });
        mapping->map(mem_handle);
        HGresult release_result = hgMemRelease(mem_handle);
        release_handle.release();
        MOONEP_HGCHECK(release_result);
    }

    std::vector<int64_t> full_shape = chunk_shape;
    full_shape[0] *= world_size;

    // The captured shared_ptr releases the mapping when the Storage is destroyed.
    return ptg_make_vmm_tensor(
        reinterpret_cast<void *>(mapping->ptr), chunk_nbytes * world_size,
        full_shape, dtype, device_id, [mapping](void *) {});
}
