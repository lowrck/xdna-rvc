#include "util/thread_priority.h"

#include <cerrno>
#include <cstring>
#include <stdexcept>

#if defined(_WIN32)
#include <windows.h>
#include <avrt.h>
#pragma comment(lib, "avrt.lib")
#else
#include <pthread.h>
#include <sched.h>
#include <sys/resource.h>
#include <unistd.h>
#if defined(__linux__)
#include <sys/syscall.h>
#endif
#endif

namespace xr {

const char* to_string(ThreadPriority p) {
    switch (p) {
        case ThreadPriority::Normal: return "normal";
        case ThreadPriority::High: return "high";
        case ThreadPriority::Realtime: return "realtime";
    }
    return "normal";
}

ThreadPriority parse_thread_priority(const std::string& t) {
    if (t == "normal") return ThreadPriority::Normal;
    if (t == "high") return ThreadPriority::High;
    if (t == "realtime") return ThreadPriority::Realtime;
    throw std::invalid_argument("thread priority must be normal|high|realtime");
}

std::string set_current_thread_priority(ThreadPriority p) {
    if (p == ThreadPriority::Normal) return "normal priority";
#if defined(_WIN32)
    std::string what;
    if (p == ThreadPriority::Realtime) {
        DWORD task_index = 0;
        if (AvSetMmThreadCharacteristicsW(L"Pro Audio", &task_index)) {
            what = "MMCSS 'Pro Audio'";
        } else {
            what = "MMCSS unavailable (error " + std::to_string(GetLastError()) + ")";
        }
    }
    const int prio = p == ThreadPriority::Realtime ? THREAD_PRIORITY_TIME_CRITICAL : THREAD_PRIORITY_HIGHEST;
    if (SetThreadPriority(GetCurrentThread(), prio)) {
        what += std::string(what.empty() ? "" : " + ") +
                (p == ThreadPriority::Realtime ? "THREAD_PRIORITY_TIME_CRITICAL" : "THREAD_PRIORITY_HIGHEST");
    }
    return what.empty() ? "unchanged (SetThreadPriority failed)" : what;
#else
    if (p == ThreadPriority::Realtime) {
        sched_param sp{};
        sp.sched_priority = 40;
        const int rc = pthread_setschedparam(pthread_self(), SCHED_FIFO, &sp);
        if (rc == 0) return "SCHED_FIFO priority 40";
    }
#if defined(__linux__)
    const pid_t tid = static_cast<pid_t>(syscall(SYS_gettid));
    for (int nice : {-15, -10, -5}) {
        if (setpriority(PRIO_PROCESS, static_cast<id_t>(tid), nice) == 0) {
            return "nice " + std::to_string(nice) + (p == ThreadPriority::Realtime ? " (SCHED_FIFO not permitted)" : "");
        }
    }
#endif
    return std::string("unchanged (no permission: ") + std::strerror(errno) + ")";
#endif
}

}  // namespace xr
