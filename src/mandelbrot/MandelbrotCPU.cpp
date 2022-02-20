#include <mandelbrot/MandelbrotCPU.h>

#include <cmath>
#include <stdexcept>

namespace mandelbrot {

MandelbrotCPU::MandelbrotCPU(Config const& cfg) : Mandelbrot(cfg) {}

void MandelbrotCPU::iterate(Viewport const& viewport) {
  if (m_cfg.smooth != 1)
    throw std::runtime_error("In cpu mode, smooth must be equal to 1!");
  Real boiloutRadius2 = m_boiloutRadius * m_boiloutRadius;

  Real* iterations = m_results.cpu_data();

  for (size_t y = 0; y < size_t(m_cfg.resY); ++y)
    for (size_t x = 0; x < size_t(m_cfg.resX); ++x, ++iterations) {
      Real zr = 0.0, zi = 0.0;
      Real cr = static_cast<Real>(
          viewport.x1 + (viewport.x2 - viewport.x1) *
                            (static_cast<long double>(x) /
                             static_cast<long double>(m_cfg.resX - 1)));
      Real ci = static_cast<Real>(
          viewport.y1 + (viewport.y2 - viewport.y1) *
                            (static_cast<long double>(y) /
                             static_cast<long double>(m_cfg.resY - 1)));

      *iterations = -1.0;
      size_t n = 1;
      do {
        Mandelbrot::Real zr2 = (zr) * (zr);
        Mandelbrot::Real zi2 = (zi) * (zi);

        if (zr2 + zi2 > boiloutRadius2) {
          *iterations =
              static_cast<Real>(n) -
              std::log(0.5 * std::log(zr2 + zi2) / std::log(m_boiloutRadius)) /
                  std::log(2.0);
          break;
        }

        zi = 2.0 * (zr) * (zi) + ci;
        zr = zr2 - zi2 + cr;
        ++n;
      } while (n < m_cfg.maxIters);
    }
}

void MandelbrotCPU::colorize() {
  Real const* iterations = m_results.const_cpu_data();
  float const* palette = m_palette.const_cpu_data();
  uint8_t* image = m_image.cpu_data();

  for (size_t i = 0; i < m_cfg.resX * m_cfg.resY; ++i, ++iterations) {
    if (*iterations < 0) {
      *image++ = 0;
      *image++ = 0;
      *image++ = 0;
      *image++ = 255;
    } else {
      auto index = static_cast<double>(*iterations / m_palette_scale *
                                       static_cast<Real>(m_palette_size));

      size_t ix1 = static_cast<size_t>(floor(index)) % m_palette_size;
      size_t ix2 = (ix1 + 1) % m_palette_size;

      double remainder = fmod(index, 1);

      *image++ = static_cast<uint8_t>(
          palette[3 * ix1 + 0] +
          remainder * (palette[3 * ix2 + 0] - palette[3 * ix1 + 0]));
      *image++ = static_cast<uint8_t>(
          palette[3 * ix1 + 1] +
          remainder * (palette[3 * ix2 + 1] - palette[3 * ix1 + 1]));
      *image++ = static_cast<uint8_t>(
          palette[3 * ix1 + 2] +
          remainder * (palette[3 * ix2 + 2] - palette[3 * ix1 + 2]));
      *image++ = 255;
    }
  }
}

}  // namespace mandelbrot