#ifdef WITH_CUDA

#include <mandelbrot/MandelbrotGPU.h>

namespace mandelbrot {

MandelbrotGPU::MandelbrotGPU(Config const& cfg, int deviceId)
    : Mandelbrot(cfg), m_deviceId(deviceId) {
  cudaGetDeviceProperties(&m_deviceProp, m_deviceId);
}

}  // namespace mandelbrot

#endif // WITH_CUDA
