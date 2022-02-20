#pragma once

#include <chrono>
#include <condition_variable>
#include <mutex>
#include <queue>
#include <thread>

// Thread safe queue inspired by Caffe's BlockingQueue
 
template <template <typename...> class Queue, typename T>
class BlockingQueueTemplate {
 public:
  void push(T const& v) {
    {
      std::unique_lock<std::mutex> lock(m_mutex);
      m_queue.push(v);
    }
    m_condition.notify_one();
  }

  T pop() {
    std::unique_lock<std::mutex> lock(m_mutex);
    m_condition.wait(lock, [this] { return !m_queue.empty(); });
    T v = m_queue.front();
    m_queue.pop();
    return v;
  }

  T const& front() { return m_queue.front(); }

  bool pop(T& v, int wait_ms) {
    std::unique_lock<std::mutex> lock(m_mutex);
    if (m_condition.wait_for(lock, std::chrono::milliseconds(wait_ms),
                             [this] { return !m_queue.empty(); })) {
      v = m_queue.front();
      m_queue.pop();
      return true;
    }
    return false;
  }

  size_t size() {
    std::unique_lock<std::mutex> lock(m_mutex);
    return m_queue.size();
  }

 private:
  Queue<T> m_queue;
  std::mutex m_mutex;
  std::condition_variable m_condition;
};

template <typename T>
class PriorityQueueAdapter
    : public std::priority_queue<T, std::vector<T>, std::greater<T>> {
 public:
  T const& front() const {
    return std::priority_queue<T, std::vector<T>, std::greater<T>>::top();
  }
};

template <typename T>
using BlockingQueue = BlockingQueueTemplate<std::queue, T>;

template <typename T>
using BlockingPriorityQueue = BlockingQueueTemplate<PriorityQueueAdapter, T>;
