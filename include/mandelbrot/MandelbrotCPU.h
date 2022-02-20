#pragma once

#include <mandelbrot/Mandelbrot.h>

namespace mandelbrot {

class MandelbrotCPU : public Mandelbrot {
 public:
  MandelbrotCPU(Config const& cfg);

  void iterate(Viewport const& viewport) override;
  void colorize() override;
};

} // namespace mandelbrot