#include <utils/GIFWriter.h>

#include <cstring>
#include <filesystem>
#include <iostream>

GIFWriter::GIFWriter(Config const& cfg) : m_cfg(cfg), m_running(false) {}

GIFWriter::~GIFWriter() {
  if(m_running)
  {
    m_running = false;
    m_thread.join();
    GifEnd(&m_writer);
  }
}

void GIFWriter::open(size_t resX, size_t resY) {
  m_resX = resX;
  m_resY = resY;

  std::filesystem::path p(m_cfg.filename);
  std::filesystem::create_directories(p.parent_path());

  GifBegin(&m_writer, m_cfg.filename.c_str(), static_cast<uint32_t>(m_resX),
           static_cast<uint32_t>(m_resY), m_cfg.delay);

  for (size_t i = 0; i < m_cfg.startPoolSize; ++i) {
    m_free.push(std::make_shared<Image>(imageDataLength(), 0));
  }
  m_poolSize = m_cfg.startPoolSize;
  
  m_running = true;
  m_thread = std::thread(&GIFWriter::worker, this);
}

void GIFWriter::push(Index index, uint8_t const* data) {
  SPImage im;

  if (0 == m_free.size()) {
    std::unique_lock<std::mutex> lock(m_mutex);
    if (0 == m_free.size() && m_poolSize < m_cfg.maxPoolSize) {
      m_free.push(std::make_shared<Image>(data, data + imageDataLength()));
      ++m_poolSize;
    }
  }
  im = m_free.pop();
  memcpy(im->data(), data, imageDataLength());
  m_full.push(std::make_pair(index, im));
}

void GIFWriter::worker() {
  Index nextFrameIndex = 0;

  while (true) {
    if (!m_running && 0 == m_full.size()) return;

    if (0 == m_full.size()) {
      std::this_thread::sleep_for(std::chrono::milliseconds(200));
      continue;
    }

    if (m_full.front().first != nextFrameIndex) {
      if (m_cfg.maxPoolSize == m_full.size())
        throw std::runtime_error(
            "GIFWriter has reached maxPoolSize without receiving the next "
            "frame!");

      std::this_thread::sleep_for(std::chrono::milliseconds(200));
      continue;
    }

    IndexedSPImage im;

    if (m_full.pop(im, 200)) {
      GifWriteFrame(&m_writer, im.second->data(), static_cast<uint32_t>(m_resX),
                    static_cast<uint32_t>(m_resY), m_cfg.delay);
      if (m_poolSize > 0) m_free.push(im.second);
      ++nextFrameIndex;
    }
  }
}

size_t GIFWriter::imageDataLength() const { return 4 * m_resX * m_resY; }
