#include "engine/inference_control.h"
#include <iostream>
#include <stdexcept>

namespace {
using lighton::detail::Control;
using lighton::detail::Clock;
void require(bool value, const char* message) {
  if (!value) throw std::runtime_error(message);
}
void expect_error(Control& control, lighton::EngineError::Kind kind) {
  try {
    control.check();
  } catch (const lighton::EngineError& error) {
    require(error.kind == kind, "wrong stop classification");
    return;
  }
  throw std::runtime_error("stopped control did not throw");
}
}

int main() {
  try {
    unsigned expired_polls = 0;
    Control expired{Clock::time_point::min(), [&] { ++expired_polls; return false; }};
    require(expired.stop(), "expired control must abort native work");
    expect_error(expired, lighton::EngineError::deadline);
    require(expired_polls == 0, "expired deadline must not depend on transport polling");

    unsigned polls = 0;
    bool cancelled = false;
    Control active{Clock::time_point::max(), [&] { ++polls; return cancelled; }};
    active.check();
    require(polls == 1, "live control polls cancellation");
    // Pin the cache boundary explicitly: no sleeps or wall-clock assumptions.
    active.next_poll = Clock::time_point::max();
    active.check();
    require(!active.stop() && polls == 1, "live polling cache must remain effective");
    active.deadline = Clock::time_point::min();
    expect_error(active, lighton::EngineError::deadline);
    require(polls == 1, "deadline must bypass an unexpired polling cache");

    unsigned cancellation_polls = 0;
    Control client{Clock::time_point::max(), [&] { ++cancellation_polls; return true; }};
    expect_error(client, lighton::EngineError::cancelled);
    require(client.stop() && cancellation_polls == 1, "client cancellation must latch");
    client.deadline = Clock::time_point::min();
    expect_error(client, lighton::EngineError::cancelled);
    require(cancellation_polls == 1, "observed client cancellation must not be re-polled");
    std::cout << "PASS: expired deadline, cached polling and real cancellation classification\n";
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
