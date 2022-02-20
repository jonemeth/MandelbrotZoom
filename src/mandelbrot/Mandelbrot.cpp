//
// Created by jnemeth on 2/5/22.
//

#include <mandelbrot/Mandelbrot.h>
#include <utils/Palette.h>

namespace mandelbrot {

Mandelbrot::Mandelbrot(Config const& cfg)
    : m_cfg(cfg),
      m_results(m_cfg.resX * m_cfg.resY),
      m_image(4 * m_cfg.resX * m_cfg.resY),
      m_palette(defaultPalette()),
      m_palette_size(m_palette.size() / 3),
      m_palette_scale(256.0),
      m_boiloutRadius(4.0) {}

void Mandelbrot::render(Viewport const& viewport) {
  iterate(viewport);
  colorize();
}

}  // namespace mandelbrot