#pragma once

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <tuple>
#include <vector>

#include <hggc.h>

#include <ATen/ATen.h>
#include <ATen/DeviceAccelerator.h>
#include <c10/core/DeviceGuard.h>

#ifndef HGCHECK
#define HGCHECK(cmd) do {                                      \
    HGresult err = cmd;                                            \
    if (err != HGGC_SUCCESS) {                                     \
        const char *errStr;                                        \
        hgGetErrorString(err, &errStr);                            \
        fprintf(stderr, "Failed: HG error %s:%d '%s'\n",          \
            __FILE__, __LINE__, errStr);                           \
        exit(EXIT_FAILURE);                                        \
    }                                                              \
} while(0)
#endif

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

static inline bool icn_fabric_supported() {
    const int device_id = ptg_current_device();
    HGdevice hg_device;
    HGCHECK(hgDeviceGet(&hg_device, device_id));
    int supported = 0;
    HGresult err = hgDeviceGetAttribute(&supported,
        HG_DEVICE_ATTRIBUTE_HANDLE_TYPE_FABRIC_SUPPORTED, hg_device);
    // Older drivers do not know the attribute at all.
    if (err != HGGC_SUCCESS || !supported) return false;

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
    HGCHECK(hgMemRelease(handle));
    return true;
}

static inline size_t icn_granularity_for(int device_id,
                                         HGmemAllocationHandleType ht) {
    HGmemAllocationProp prop = {};
    prop.type = HG_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type = HG_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id = device_id;
    prop.requestedHandleTypes = ht;

    size_t granularity;
    HGCHECK(hgMemGetAllocationGranularity(
        &granularity, &prop, HG_MEM_ALLOC_GRANULARITY_RECOMMENDED));
    return granularity;
}

static inline size_t icn_granularity_max(int device_id) {
    size_t gran = icn_granularity_for(
        device_id, HG_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR);
    if (icn_fabric_supported()) {
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

static inline at::Tensor make_vmm_tensor(
    void *ptr,
    size_t nbytes,
    size_t allocated_size,
    int device_id,
    const std::vector<int64_t> &shape,
    at::ScalarType dtype
) {
    auto deleter = [device_id, ptr, allocated_size](void *p) {
        if (!p) return;
        c10::DeviceGuard guard(c10::Device(ptg_device_type(), device_id));
        ptg_synchronize(device_id);
        hgMemUnmap(reinterpret_cast<HGdeviceptr>(ptr), allocated_size);
        hgMemAddressFree(reinterpret_cast<HGdeviceptr>(ptr), allocated_size);
    };
    return ptg_make_vmm_tensor(
        ptr, nbytes, shape, dtype, device_id, std::move(deleter));
}

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
        HGCHECK(hgMemExportToShareableHandle(
            &fabric, handle, HG_MEM_HANDLE_TYPE_FABRIC, 0));
        auto out = at::empty({kFabricHandleBytes},
                             at::TensorOptions().dtype(at::kByte));
        std::memcpy(out.data_ptr(), fabric.data, sizeof(fabric.data));
        return out;
    }
    int fd = -1;
    HGCHECK(hgMemExportToShareableHandle(
        &fd, handle, HG_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR, 0));
    TORCH_CHECK(fd >= 0, "hgMemExportToShareableHandle returned invalid fd");
    return at::scalar_tensor(fd, at::TensorOptions().dtype(at::kLong));
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
        HGCHECK(hgMemImportFromShareableHandle(
            &handle, &fabric, HG_MEM_HANDLE_TYPE_FABRIC));
    } else {
        int64_t fd = handles.const_data_ptr<int64_t>()[index];
        TORCH_CHECK(fd >= 0, "invalid fd ", fd, " at index ", index);
        HGCHECK(hgMemImportFromShareableHandle(
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
    HGCHECK(hgMemCreate(&mem_handle, allocated_size, &prop, 0));

    HGdeviceptr dptr;
    HGCHECK(hgMemAddressReserve(&dptr, allocated_size, 0, 0, 0));
    HGCHECK(hgMemMap(dptr, allocated_size, 0, mem_handle, 0));

    HGmemAccessDesc access_desc = {};
    access_desc.location.type = HG_MEM_LOCATION_TYPE_DEVICE;
    access_desc.location.id = device_id;
    access_desc.flags = HG_MEM_ACCESS_FLAGS_PROT_READWRITE;
    HGCHECK(hgMemSetAccess(dptr, allocated_size, &access_desc, 1));

    // Export for other processes to import. The caller releases mem_handle
    // after every required mapping has been established.
    auto shareable = icn_export_shareable(mem_handle, use_fabric);

    auto keepalive = make_vmm_tensor(
        reinterpret_cast<void *>(dptr), nbytes, allocated_size,
        device_id, chunk_shape, dtype);

    return {std::move(keepalive), std::move(shareable),
            static_cast<int64_t>(mem_handle)};
}

// Release the owned mem handle returned by icn_dist_alloc (the physical memory
// stays alive via the existing maps).
static inline void icn_release_mem_handle(int64_t mem_handle_u64) {
    HGCHECK(hgMemRelease(static_cast<HGmemGenericAllocationHandle>(mem_handle_u64)));
}

/**
 * Map all chunk handles into a contiguous VA region.
 * Key difference from reference: ALL chunks get RW access (not just local_rank),
 * because dispatch needs to write to remote ranks' regions.
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
    icn_check_handles(shareables, world_size, use_fabric);

    auto [chunk_nbytes, chunk_allocated_size, device_id] =
        icn_prepare(chunk_shape, dtype, use_fabric);

    TORCH_CHECK(chunk_allocated_size == chunk_nbytes,
        "Chunk byte size (", chunk_nbytes, ") must be aligned to VMM granularity (",
        chunk_allocated_size, "). Adjust chunk dimensions so that the total bytes "
        "are a multiple of the allocation granularity.");

    size_t total_allocated_size = chunk_allocated_size * world_size;

    HGdeviceptr dptr;
    HGCHECK(hgMemAddressReserve(&dptr, total_allocated_size, 0, 0, 0));

    for (int64_t i = 0; i < world_size; i++) {
        HGmemGenericAllocationHandle mem_handle =
            icn_import_shareable(shareables, i, use_fabric);

        HGdeviceptr chunk_va = dptr + i * chunk_allocated_size;
        HGCHECK(hgMemMap(chunk_va, chunk_allocated_size, 0, mem_handle, 0));

        // All chunks get RW access for dispatch writes
        HGmemAccessDesc access = {};
        access.location.type = HG_MEM_LOCATION_TYPE_DEVICE;
        access.location.id = device_id;
        access.flags = HG_MEM_ACCESS_FLAGS_PROT_READWRITE;
        HGCHECK(hgMemSetAccess(chunk_va, chunk_allocated_size, &access, 1));

        HGCHECK(hgMemRelease(mem_handle));
    }

    std::vector<int64_t> full_shape = chunk_shape;
    full_shape[0] *= world_size;

    int ws = world_size;
    size_t cas = chunk_allocated_size;
    size_t tas = total_allocated_size;
    auto deleter = [device_id, dptr, cas, ws, tas](void *p) {
        if (!p) return;
        c10::DeviceGuard guard(c10::Device(ptg_device_type(), device_id));
        ptg_synchronize(device_id);
        for (int i = 0; i < ws; i++) {
            hgMemUnmap(dptr + i * cas, cas);
        }
        hgMemAddressFree(dptr, tas);
    };

    return ptg_make_vmm_tensor(
        reinterpret_cast<void *>(dptr), chunk_nbytes * world_size,
        full_shape, dtype, device_id, std::move(deleter));
}
