#include "microtorch/device.hpp"

#include <cstdlib>
#include <cstring>
#include <stdexcept>

#ifdef MICROTORCH_CUDA
#include "cuda_kernels.hpp"   // transformer_core: CudaMatrix
#endif

namespace microtorch {
namespace device {

namespace {
Device g_device = Device::CPU;
}

Device get() { return g_device; }

bool cuda_compiled() {
#ifdef MICROTORCH_CUDA
    return true;
#else
    return false;
#endif
}

void set(Device d) {
    if (d == Device::CUDA && !cuda_compiled()) {
        throw std::runtime_error(
            "device::set(CUDA): built without -DMICROTORCH_CUDA=ON");
    }
    g_device = d;
}

void set_from_env() {
    const char* v = std::getenv("MICROTORCH_DEVICE");
    if (!v) return;
    if (std::strcmp(v, "cuda") == 0) set(Device::CUDA);
    else if (std::strcmp(v, "cpu") == 0) set(Device::CPU);
    else throw std::runtime_error("MICROTORCH_DEVICE must be cpu or cuda");
}

Matrix matmul(const Matrix& a, const Matrix& b) {
#ifdef MICROTORCH_CUDA
    if (g_device == Device::CUDA) {
        // Phase A: host-resident tensors, per-call round trip. Correctness
        // is gradcheck-gated (the same suite runs under either device).
        CudaMatrix da(a), db(b);
        return CudaMatrix::matmul(da, db).to_host();
    }
#endif
    return matmul_optimized(a, b);
}

}  // namespace device
}  // namespace microtorch
