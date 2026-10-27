#include <torch/extension.h>
#include <ATen/DeviceAccelerator.h>
#include "ppu_shared_buffer.hpp"

/**
 * Return the VMM allocation granularity in bytes for the current device.
 */
int64_t get_vmm_granularity() {
    const int device_id = at::accelerator::getDeviceIndex();
    return static_cast<int64_t>(icn_granularity_max(device_id));
}

PYBIND11_MODULE(_C, m) {
    m.attr("TILELANG_PPU_BUILD") = true;
    m.attr("FABRIC_HANDLE_BYTES") = kFabricHandleBytes;
    m.def("icn_dist_alloc", &icn_dist_alloc,
          pybind11::arg("shape"), pybind11::arg("dtype"),
          pybind11::arg("use_fabric") = false,
          "Allocate a local chunk and export it for sharing, as a POSIX fd "
          "(same node) or a 64-byte fabric handle (ICN).");
    m.def("icn_dist_map", &icn_dist_map,
          pybind11::arg("chunk_shape"), pybind11::arg("dtype"),
          pybind11::arg("shareables"), pybind11::arg("local_rank"),
          pybind11::arg("world_size"), pybind11::arg("use_fabric") = false,
          "Map all ranks' chunks into a contiguous VA region (all RW).");
    m.def("icn_fabric_supported", &icn_fabric_supported,
          "Whether the current device can export/import fabric handles.");
    m.def("get_vmm_granularity", &get_vmm_granularity,
          "Return VMM allocation granularity in bytes.");
    m.def("icn_release_mem_handle", &icn_release_mem_handle,
          pybind11::arg("mem_handle"),
          "Release an owned mem handle returned by icn_dist_alloc.");
}
