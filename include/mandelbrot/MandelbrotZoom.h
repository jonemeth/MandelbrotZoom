#pragma once

#include <mandelbrot/Mandelbrot.h>
#include <utils/ImageWriter.h>

#include <mutex>
#include <thread>


namespace mandelbrot {


class MandelbrotZoom {
 public:
  struct Config {
    Device device;
    Point center;
    Viewport baseViewport;
    long double scaleStep;
    long double minScale;
    size_t numThreads;
  };

  struct ZoomStep {
    size_t iter;
    long double scale;
  };

  using Writers = std::vector<ImageWriter::SP>;

 public:
  MandelbrotZoom(Config const& cfg, Mandelbrot::Config const& mandelbrotCfg,
                 Writers writer);
  void run();

 private:
  void worker(int ix);
  bool nextStep(ZoomStep& step);
  Viewport getViewport(ZoomStep const& step) const;

 private:
  Config m_cfg;
  Mandelbrot::Config m_mandelbrotCfg;
  Writers m_writers;
  std::vector<std::thread> m_threads;

  std::mutex m_step_mutex;
  ZoomStep m_step;
};


}  // namespace mandelbrot