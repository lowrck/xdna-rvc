#pragma once

#include <string>

namespace xr {

enum class ThreadPriority { Normal, High, Realtime };

// Raises the calling thread's scheduling priority. Windows: MMCSS "Pro Audio" task +
// SetThreadPriority. Linux: SCHED_FIFO if permitted (rtprio limit), else a lower nice
// value, else unchanged. Returns a description of what was actually applied (never throws).
std::string set_current_thread_priority(ThreadPriority priority);

ThreadPriority parse_thread_priority(const std::string& text);  // normal|high|realtime
const char* to_string(ThreadPriority p);

}  // namespace xr
