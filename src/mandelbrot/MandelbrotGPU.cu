#ifdef WITH_CUDA

#include <mandelbrot/MandelbrotGPU.h>

namespace mandelbrot {

#define MAX_BLOCK_SIZE (32u)

// Indexing shared memory
#define shIx(a, b) ((b)*blockDim.x + (a))

template <typename T>
__device__ void inline swap(T &a, T &b) {
  T c(a);
  a = b;
  b = c;
}

__global__ void iterate_kernel(unsigned int gridSizeX, unsigned int gridSizeY,
                               unsigned int smooth, Mandelbrot::Real minX,
                               Mandelbrot::Real maxX, Mandelbrot::Real minY,
                               Mandelbrot::Real maxY, size_t maxIters,
                               Mandelbrot::Real *results,
                               Mandelbrot::Real boiloutRadius) {
  extern __shared__ Mandelbrot::Real blockResults[];

  unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= gridSizeX || y >= gridSizeY) return;

  results += (y / smooth) * (gridSizeX / smooth) + (x / smooth);

  Mandelbrot::Real boiloutRadius2 = boiloutRadius * boiloutRadius;

  Mandelbrot::Real zr = 0.0;
  Mandelbrot::Real zi = 0.0;
  Mandelbrot::Real cr =
      minX + (maxX - minX) * (((Mandelbrot::Real)(x)) /
                              ((Mandelbrot::Real)(gridSizeX - 1)));
  Mandelbrot::Real ci =
      minY + (maxY - minY) * (((Mandelbrot::Real)(y)) /
                              ((Mandelbrot::Real)(gridSizeY - 1)));

  blockResults[shIx(threadIdx.x, threadIdx.y)] = -1;
  int n = 1;

  do {
    Mandelbrot::Real zr2 = (zr) * (zr);
    Mandelbrot::Real zi2 = (zi) * (zi);

    if (zr2 + zi2 > boiloutRadius2) {
      // https://en.wikipedia.org/wiki/Plotting_algorithms_for_the_Mandelbrot_set
      Mandelbrot::Real result =
          n - log(0.5 * log(zr2 + zi2) / log(boiloutRadius)) / log(2.0);

      if (1 == smooth) {
        *results = result;
        return;
      }

      blockResults[shIx(threadIdx.x, threadIdx.y)] = result;

      break;
    }

    zi = 2.0 * (zr) * (zi) + ci;
    zr = zr2 - zi2 + cr;
    ++n;
  } while (n <= maxIters);

  __syncthreads();
  if (!(0 == x % smooth && 0 == y % smooth)) return;

  int numVals = 0;
  Mandelbrot::Real vals[MAX_BLOCK_SIZE * MAX_BLOCK_SIZE];

  for (unsigned int a = threadIdx.x; a < threadIdx.x + smooth; ++a)
    for (unsigned int b = threadIdx.y; b < threadIdx.y + smooth; ++b)
      if (blockResults[shIx(a, b)] != -1.0) {
        vals[numVals] = blockResults[shIx(a, b)];
        ++numVals;
      }

  if (numVals < smooth * smooth / 2) {
    *results = -1.0;
    return;
  }

  int ix1 = 0, ix2 = numVals - 1;

  if (numVals >= 4) {
    // Bubble sort vals
    for (int i = 0; i < numVals - 1; ++i)
      for (int j = i + 1; j < numVals; ++j)
        if (vals[j] > vals[i]) swap(vals[j], vals[i]);

    // Discard the first and last quadrants
    ix1 = numVals / 4;
    ix2 = (3 * numVals) / 4;
  }

  // Harmonic mean
  Mandelbrot::Real sum = 0.0;
  for (int i = ix1; i <= ix2; ++i) sum += 1.0 / vals[i];
  *results = 1.0 / (sum / (1 + ix2 - ix1));
}

void MandelbrotGPU::iterate(Viewport const &viewport) {
  auto deviceMaxBlockSize =
      static_cast<unsigned int>(std::sqrt(m_deviceProp.maxThreadsPerBlock));
  
  auto maxBlockSize = std::min(MAX_BLOCK_SIZE, deviceMaxBlockSize);
  
  unsigned int smooth =
      std::min(maxBlockSize, static_cast<unsigned int>(m_cfg.smooth));

  auto blockSize = static_cast<unsigned int>(maxBlockSize / smooth) * smooth;
  dim3 block_dim{blockSize, blockSize};

  unsigned int gridSizeX = m_cfg.resX * smooth;
  unsigned int gridSizeY = m_cfg.resY * smooth;
  dim3 grid_size{(gridSizeX + block_dim.x - 1) / block_dim.x,
                 (gridSizeY + block_dim.y - 1) / block_dim.y};

  size_t shared_size = block_dim.x * block_dim.y * sizeof(Mandelbrot::Real);
  iterate_kernel<<<grid_size, block_dim, shared_size>>>(
      gridSizeX, gridSizeY, smooth, viewport.x1, viewport.x2, viewport.y1,
      viewport.y2, m_cfg.maxIters, m_results.gpu_data(), m_boiloutRadius);
}

__global__ void colorize_kernel(size_t size, Mandelbrot::Real const *iterations,
                                float const *palette, size_t palette_size,
                                Mandelbrot::Real palette_scale,
                                uint8_t *image) {
  unsigned int ix = blockIdx.x * blockDim.x + threadIdx.x;
  if (ix >= size) return;

  iterations += ix;
  image += 4 * ix;

  if (*iterations < 0) {
    *image++ = 0;
    *image++ = 0;
    *image++ = 0;
    *image = 255;
  } else {
    auto index = static_cast<float>(*iterations / palette_scale *
                                    static_cast<double>(palette_size));

    size_t ix1 = static_cast<size_t>(floor(index)) % palette_size;
    size_t ix2 = (ix1 + 1) % palette_size;

    float remainder = fmodf(index, 1);

    *image++ = static_cast<uint8_t>(
        palette[3 * ix1 + 0] +
        remainder * (palette[3 * ix2 + 0] - palette[3 * ix1 + 0]));
    *image++ = static_cast<uint8_t>(
        palette[3 * ix1 + 1] +
        remainder * (palette[3 * ix2 + 1] - palette[3 * ix1 + 1]));
    *image++ = static_cast<uint8_t>(
        palette[3 * ix1 + 2] +
        remainder * (palette[3 * ix2 + 2] - palette[3 * ix1 + 2]));
    *image = 255;
  }
}

void MandelbrotGPU::colorize() {
  size_t block_size = m_deviceProp.maxThreadsPerBlock;
  size_t size = m_cfg.resX * m_cfg.resY;
  size_t grid_size = (size + block_size - 1) / block_size;
  colorize_kernel<<<grid_size, block_size>>>(
      size, m_results.const_gpu_data(), m_palette.const_gpu_data(),
      m_palette_size, m_palette_scale, m_image.gpu_data());
}

}  // namespace mandelbrot

#endif  // WITH_CUDA
