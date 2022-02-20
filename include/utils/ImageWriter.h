
#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>

class ImageWriter {
 public:
  using SP = std::shared_ptr<ImageWriter>;
  using Index = size_t;

 public:
  virtual void open(size_t resX, size_t resY) = 0;
  virtual void push(Index ix, uint8_t const* data) = 0;
};
