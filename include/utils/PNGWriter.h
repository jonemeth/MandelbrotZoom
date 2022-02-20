
#pragma once
#include <utils/BlockingQueue.h>
#include <utils/ImageWriter.h>

#include <atomic>

class PNGWriter : public ImageWriter {
 public:
  struct Config {
    std::string outputPath;
    size_t startPoolSize;
    size_t maxPoolSize;
    size_t numThreads;
  };

 private:
  using Image = std::vector<uint8_t>;
  using SPImage = std::shared_ptr<Image>;
  using IndexedSPImage = std::pair<Index, SPImage>;

 public:
  PNGWriter(Config const& cfg);
  ~PNGWriter();

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

  std::mutex m_mutex;
  size_t m_poolSize;

  BlockingQueue<SPImage> m_free;
  BlockingQueue<IndexedSPImage> m_full;

  std::vector<std::thread> m_threads;
};
