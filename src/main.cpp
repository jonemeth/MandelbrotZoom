#include <mandelbrot/MandelbrotZoom.h>
#include <thirdparty/json.h>
#include <utils/GIFWriter.h>
#include <utils/PNGWriter.h>

#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>

static std::string trimJSONString(std::string const& str) {
  assert(str.front() == '"' && str.back() == '"');
  return str.substr(1, str.size() - 2);
}

mandelbrot::Mandelbrot::Config getMandelbrotConfig(json::jobject j) {
  mandelbrot::Mandelbrot::Config cfg;
  cfg.resX = std::stoi(j.get("resX"));
  cfg.resY = std::stoi(j.get("resY"));
  cfg.smooth = std::stoi(j.get("smooth"));
  cfg.maxIters = std::stoi(j.get("maxIters"));
  return cfg;
}

Point getPoint(json::jobject j) {
  Point p;
  p.x = std::stold(j.get("x"));
  p.y = std::stold(j.get("y"));
  return p;
}

Viewport getViewport(json::jobject j) {
  Viewport vp;
  vp.x1 = std::stold(j.get("x1"));
  vp.x2 = std::stold(j.get("x2"));
  vp.y1 = std::stold(j.get("y1"));
  vp.y2 = std::stold(j.get("y2"));
  return vp;
}

mandelbrot::MandelbrotZoom::Config getMandelbrotZoomConfig(json::jobject j) {
  mandelbrot::MandelbrotZoom::Config cfg;

  std::string device = j.get("device");
  if (device == "\"cpu\"")
    cfg.device = Device::CPU;
  else if (device == "\"gpu\"")
    cfg.device = Device::GPU;
  else
    throw std::runtime_error("Unknown device!");

  cfg.center = getPoint(j["center"].as_object());
  cfg.baseViewport = getViewport(j["baseViewport"].as_object());
  cfg.scaleStep = std::stold(j.get("scaleStep"));
  cfg.minScale = std::stold(j.get("minScale"));
  cfg.numThreads = std::stoi(j.get("numThreads"));
  return cfg;
}

GIFWriter::Config getGIFWriterConfig(json::jobject j) {
  GIFWriter::Config cfg;
  cfg.filename = trimJSONString(j.get("filename"));
  cfg.delay = std::stoi(j.get("delay"));
  cfg.startPoolSize = std::stoi(j.get("startPoolSize"));
  cfg.maxPoolSize = std::stoi(j.get("maxPoolSize"));
  return cfg;
}

PNGWriter::Config getPNGWriterConfig(json::jobject j) {
  PNGWriter::Config cfg;
  cfg.outputPath = trimJSONString(j.get("outputPath"));
  cfg.startPoolSize = std::stoi(j.get("startPoolSize"));
  cfg.maxPoolSize = std::stoi(j.get("maxPoolSize"));
  cfg.numThreads = std::stoi(j.get("numThreads"));
  return cfg;
}

mandelbrot::MandelbrotZoom::Writers getWriters(json::jobject j) {
  mandelbrot::MandelbrotZoom::Writers writers;
  if (j.has_key("GIFWriter")) {
    GIFWriter::Config cfg = getGIFWriterConfig(j["GIFWriter"].as_object());
    writers.push_back(std::make_shared<GIFWriter>(cfg));
  }

  if (j.has_key("PNGWriter")) {
    PNGWriter::Config cfg = getPNGWriterConfig(j["PNGWriter"].as_object());
    writers.push_back(std::make_shared<PNGWriter>(cfg));
  }
  return writers;
}

int main(int argc, char* argv[]) {
  using namespace mandelbrot;
  namespace fs = std::filesystem;

  if (argc != 2) {
    std::cout << "Usage: Mandelbrot JSON" << std::endl;
    return -1;
  }

  std::string inFile{argv[1]};

  if (!fs::exists(inFile)) {
    std::cout << "File " << inFile << " does not exist!" << std::endl;
    return -1;
  }

  if (fs::is_directory(inFile)) {
    std::cout << inFile << " is a directory!" << std::endl;
    return -1;
  }

  std::ifstream t(inFile);
  std::string str((std::istreambuf_iterator<char>(t)),
                  std::istreambuf_iterator<char>());

  json::jobject config = json::jobject::parse(str);

  Mandelbrot::Config mandelbrotCfg =
      getMandelbrotConfig(config["mandelbrot"].as_object());

  MandelbrotZoom::Config zoomCfg =
      getMandelbrotZoomConfig(config["zoom"].as_object());

  MandelbrotZoom::Writers writers = getWriters(config);

#ifdef WITH_CUDA
  std::cout << "CUDA" << std::endl;
#else
  std::cout << "NO CUDA" << std::endl;
#endif

  try {
    MandelbrotZoom zoom(zoomCfg, mandelbrotCfg, writers);
    zoom.run();
    std::cout << "Zoom ready! Waiting for image writers to finish!"
              << std::endl;

  } catch (std::exception& e) {
    std::cout << e.what() << std::endl;
  }

  return 0;
}
