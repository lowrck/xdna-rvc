#pragma once

#include <stdexcept>
#include <string>

namespace xr {

// An error intended to be shown to a user. `what()` describes what failed,
// `hint()` describes what to do about it. Both should be specific: name the model,
// stage, provider, file or device involved.
class UserError : public std::runtime_error {
public:
    UserError(std::string message, std::string hint = {})
        : std::runtime_error(std::move(message)), hint_(std::move(hint)) {}
    const std::string& hint() const { return hint_; }

private:
    std::string hint_;
};

}  // namespace xr
