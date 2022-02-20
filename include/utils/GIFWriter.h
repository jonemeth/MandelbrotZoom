#pragma once

#include <thirdparty/gif.h>
#include <utils/BlockingQueue.h>
#include <utils/ImageWriter.h>

#include <atomic>

class GIFWriter : public ImageWriter {
 public:
  struct Config {
    std::string filename;
    int32_t delay;
    size_t startPoolSize;
    size_t maxPoolSize;
  };
 private:
  using Image = std::vector<uint8_t>;
  using SPImage = std::shared_ptr<Image>;
  using IndexedSPImage = std::pair<Index, SPImage>;

 public:
  GIFWriter(Config const &cfg);
  ~GIFWriter();

  void open(size_t resX, size_t resY) override;
  void push(Index ix, uint8_t const* data) override;

 private:
  void worker();
  inline size_t imageDataLength() const;

 private:
  Config m_cfg;
  std::atomic<bool> m_running;

  size_t m_resX;
  size_t m_resY;

  GifWriter m_writer{};

  BlockingQueue<SPImage> m_free;
  BlockingPriorityQueue<IndexedSPImage> m_full;
  
  std::mutex m_mutex;
  size_t m_poolSize;

  std::thread m_thread;
};
