#pragma once

#ifdef WITH_CUDA

#include <mandelbrot/Mandelbrot.h>

namespace mandelbrot {

class MandelbrotGPU : public Mandelbrot {
 public:
  MandelbrotGPU(Config const& cfg, int deviceId);

  void iterate(Viewport const& viewport) override;
  void colorize() override;

 private:
  int m_deviceId;
  cudaDeviceProp m_deviceProp{};
};

}  // namespace mandelbrot

#endif // WITH_CUDA
