#include <thirdparty/fpng.h>
#include <utils/PNGWriter.h>

#include <cstring>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <sstream>

PNGWriter::PNGWriter(Config const& cfg) : m_cfg(cfg), m_running(false) {}

PNGWriter::~PNGWriter() {
  if (m_running) {
    m_running = false;
    for (auto& thread : m_threads) thread.join();
  }
}

void PNGWriter::open(size_t resX, size_t resY) {
  std::filesystem::create_directories(m_cfg.outputPath);

  m_resX = resX;
  m_resY = resY;

  for (size_t i = 0; i < m_cfg.startPoolSize; ++i)
    m_free.push(std::make_shared<Image>(imageDataLength(), 0));
  m_poolSize = m_cfg.startPoolSize;

  m_running = true;
  for (size_t i = 0; i < m_cfg.numThreads; ++i)
    m_threads.emplace_back(std::thread(&PNGWriter::worker, this));
}

void PNGWriter::push(Index index, uint8_t const* data) {
  if (0 == m_free.size()) {
    std::unique_lock<std::mutex> lock(m_mutex);
    if (0 == m_free.size() && m_poolSize < m_cfg.maxPoolSize) {
      m_free.push(std::make_shared<Image>(data, data + imageDataLength()));
      ++m_poolSize;
    }
  }

  SPImage im = m_free.pop();
  memcpy(im->data(), data, imageDataLength());
  m_full.push(std::make_pair(index, im));
}

void PNGWriter::worker() {
  while (true) {
    IndexedSPImage im;

    if (m_full.pop(im, 200)) {
      std::stringstream ss;
      ss << m_cfg.outputPath << "/image" << std::setfill('0') << std::setw(8) << im.first << ".png";

      fpng::fpng_encode_image_to_file(ss.str().c_str(), im.second->data(),
                                      static_cast<uint32_t>(m_resX),
                                      static_cast<uint32_t>(m_resY), 4);

      m_free.push(im.second);
      continue;
    }

    if (!m_running) return;
  }
}

size_t PNGWriter::imageDataLength() const { return 4 * m_resX * m_resY; }