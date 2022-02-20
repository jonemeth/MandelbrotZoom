
#include <mandelbrot/MandelbrotZoom.h>
#include <mandelbrot/MandelbrotGPU.h>
#include <mandelbrot/MandelbrotCPU.h>

#include <iostream>

namespace mandelbrot {

bool MandelbrotZoom::nextStep(ZoomStep& step) {
  std::unique_lock<std::mutex> lock(m_step_mutex);

  if (m_step.scale < m_cfg.minScale) return false;

  step = m_step;
  
  ++m_step.iter;
  m_step.scale *= m_cfg.scaleStep;

  return true;
}

Viewport MandelbrotZoom::getViewport(const ZoomStep& step) const {
  return {
      m_cfg.center.x + step.scale * (m_cfg.baseViewport.x1 - m_cfg.center.x),
      m_cfg.center.x + step.scale * (m_cfg.baseViewport.x2 - m_cfg.center.x),
      m_cfg.center.y + step.scale * (m_cfg.baseViewport.y1 - m_cfg.center.y),
      m_cfg.center.y + step.scale * (m_cfg.baseViewport.y2 - m_cfg.center.y)};
}

MandelbrotZoom::MandelbrotZoom(Config const& cfg,
                               Mandelbrot::Config const& mandelbrotCfg,
                               Writers writers)
    : m_cfg(cfg),
      m_mandelbrotCfg(mandelbrotCfg),
      m_writers(std::move(writers)),
      m_step{0, 1.0} {
#ifndef WITH_CUDA
  if (m_cfg.device != Device::CPU)
    throw std::runtime_error(
        "MandelbrotZoom::MandelbrotZoom: Unexpected device!");
#endif
}

void MandelbrotZoom::run() {
  for (auto& writer : m_writers)
    writer->open(m_mandelbrotCfg.resX, m_mandelbrotCfg.resY);

  for (size_t i = 0; i < m_cfg.numThreads; ++i)
    m_threads.emplace_back(std::thread(&MandelbrotZoom::worker, this, i));

  for (auto& thread : m_threads) thread.join();
  std::cout << std::endl;
}

void MandelbrotZoom::worker(int ix) {
  std::unique_ptr<Mandelbrot> m;


#ifdef WITH_CUDA
  if (m_cfg.device == Device::GPU) {
    int numDevices;
    cudaGetDeviceCount(&numDevices);
    int deviceId = ix % numDevices;
    cudaSetDevice(deviceId);
    m = std::make_unique<MandelbrotGPU>(m_mandelbrotCfg, deviceId);
  }
#endif
  if (m_cfg.device == Device::CPU)
    m = std::make_unique<MandelbrotCPU>(m_mandelbrotCfg);

  while (true) {
    ZoomStep step{};

    if (!nextStep(step)) return;

    if (0 == ix)
      std::cout << "\rit: " << step.iter << " scale: " << std::scientific
                << step.scale << std::flush;

    m->render(getViewport(step));

    for (auto& writer : m_writers) writer->push(step.iter, m->imageData());
  }
}

}  // namespace mandelbrot