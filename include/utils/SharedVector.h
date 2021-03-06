//
// Created by jnemeth on 2/5/22.
//

#pragma once

#include <string.h>

#include <cassert>
#include <cstddef>
#include <vector>

#ifdef WITH_CUDA
#include <cuda.h>
#include <cuda_runtime_api.h>
#endif

template <typename T>
class SharedVector {
 public:
  SharedVector(size_t size) : m_size(size) {}

  SharedVector(std::vector<T> const& v) : m_size(v.size()) {
    m_cpu_data = new T[m_size];
    memcpy(m_cpu_data, v.data(), m_size * sizeof(T));
    m_cpu_valid = true;
  }

  ~SharedVector() {
    if (m_cpu_data) delete[] m_cpu_data;
#ifdef WITH_CUDA
    if (m_gpu_data) cudaFree(m_gpu_data);
#endif
  }

  size_t size() const { return m_size; }

  T* cpu_data() {
    assure_cpu_valid();
#ifdef WITH_CUDA
    m_gpu_valid = false;
#endif
    return m_cpu_data;
  }

  T const* const_cpu_data() {
    assure_cpu_valid();
    return m_cpu_data;
  }

#ifdef WITH_CUDA
  T* gpu_data() {
    assure_gpu_valid();
    m_cpu_valid = false;
    return m_gpu_data;
  }
#endif

#ifdef WITH_CUDA
  T const* const_gpu_data() {
    assure_gpu_valid();
    return m_gpu_data;
  }
#endif

 private:
  void assure_cpu_valid() {
    if (!m_cpu_data) m_cpu_data = new T[m_size];
#ifdef WITH_CUDA
    if (!m_cpu_valid && m_gpu_valid) {
      cudaMemcpy(m_cpu_data, m_gpu_data, m_size * sizeof(T),
                 cudaMemcpyDeviceToHost);
    }
#endif
    m_cpu_valid = true;
  }

#ifdef WITH_CUDA
  void assure_gpu_valid() {
    if (!m_gpu_data)
      cudaMalloc(reinterpret_cast<void**>(&m_gpu_data), m_size * sizeof(T));
    if (!m_gpu_valid && m_cpu_valid) {
      cudaMemcpy(m_gpu_data, m_cpu_data, m_size * sizeof(T),
                 cudaMemcpyHostToDevice);
    }
    m_gpu_valid = true;
  }
#endif

 private:
  size_t m_size;
  T* m_cpu_data = nullptr;
  bool m_cpu_valid = false;
#ifdef WITH_CUDA
  T* m_gpu_data = nullptr;
  bool m_gpu_valid = false;
#endif
};
