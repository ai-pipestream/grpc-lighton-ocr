#pragma once

#include "engine/native_engine.h"
#include <chrono>
#include <iostream>
#include <mutex>

namespace lighton::detail {
using Clock = std::chrono::steady_clock;
struct Control {
  Clock::time_point deadline;
  std::function<bool()> cancelled;
  mutable std::mutex poll_mutex{};
  mutable Clock::time_point next_poll{};
  mutable bool cancellation_seen = false;
  mutable uint64_t polls = 0;
  mutable double polling_ms = 0;
  enum class StopReason { running, client_cancelled, deadline };
  StopReason stop_reason() const {
    std::lock_guard lock(poll_mutex);
    const auto now = Clock::now();
    if (now >= deadline || cancellation_seen)
      return cancellation_seen ? StopReason::client_cancelled : StopReason::deadline;
    // A synchronous gRPC cancellation query polls its completion queue.
    // Calling it for every graph node can dominate the model's arithmetic.
    if (now >= next_poll) {
      cancellation_seen = cancelled();
      polling_ms += std::chrono::duration<double, std::milli>(Clock::now() - now).count();
      ++polls;
      next_poll = Clock::now() + std::chrono::milliseconds(10);
    }
    return cancellation_seen ? StopReason::client_cancelled : StopReason::running;
  }
  bool stop() const { return stop_reason() != StopReason::running; }
  ~Control() {
    std::cerr << "native_phase cancel_polls=" << polls << " ms=" << polling_ms << '\n';
  }
  void check() const {
    switch (stop_reason()) {
      case StopReason::client_cancelled:
        throw EngineError(EngineError::cancelled, "client cancelled inference");
      case StopReason::deadline:
        throw EngineError(EngineError::deadline, "native inference deadline exceeded");
      case StopReason::running: return;
    }
  }
};
}  // namespace lighton::detail
