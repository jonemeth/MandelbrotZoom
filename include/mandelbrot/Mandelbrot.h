//
// Created by jnemeth on 2/5/22.
//

#pragma once

#include <mandelbrot/types.h>
#include <stdint.h>
#include <utils/SharedVector.h>

#include <vector>

namespace mandelbrot {

class Mandelbrot {
 public:
  using Real = double;

  struct Config {
    size_t resX;
    size_t resY;
    size_t smooth;
    size_t maxIters;
  };

 public:
  Mandelbrot(Config const& cfg);

  void render(Viewport const& viewport);
  uint8_t const* imageData() { return m_image.const_cpu_data(); }

  virtual void iterate(Viewport const& viewport) = 0;
  virtual void colorize() = 0;

 protected:
  Config m_cfg;

  SharedVector<Real> m_results;
  SharedVector<uint8_t> m_image;
  SharedVector<float> m_palette;
  size_t m_palette_size;
  Real m_palette_scale;
  Real m_boiloutRadius;
};

}  // namespace mandelbrot