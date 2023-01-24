/** Copyright (C) 2022  Frieder Pankratz <frieder.pankratz@gmail.com> **/

#include "Project.h"
#include <fmt/format.h>
#include <algorithm>
#include <spdlog/spdlog.h>

namespace traact::gui::state::project {
Project::Project(const std::shared_ptr<PatternFactory> &pattern_factory) : pattern_factory_(pattern_factory) {}

std::shared_ptr<Pattern> Project::addPattern(const std::string &pattern_id) {
    std::string instance_id;

    for (int i = 0; i < std::numeric_limits<int>::max(); ++i) {
        instance_id = fmt::format("pattern_{0}", i);
        auto result = findPattern(instance_id);
        if (result == patterns_.cend()) {
            break;
        }
    }

    if (instance_id.empty()) {
        SPDLOG_ERROR("could not generate new unique instance id for pattern {0}", pattern_id);
        return {};
    } else {
        auto foo = addPattern(pattern_id, instance_id);
        return foo;
    }

}

std::shared_ptr<Pattern> Project::addPattern(const std::string &pattern_id, const std::string &instance_id) {
    auto result = findPattern(instance_id);
    if (result == patterns_.cend()) {
        //patterns_.emplace_back(pattern_factory_->createPattern(pattern_id, instance_id));
        return {patterns_.back()};
    } else {
        SPDLOG_ERROR("instance id: {1}: already in use when trying to create pattern: {0}", pattern_id, instance_id);
        return {};
    }
}
std::vector<std::shared_ptr<Pattern>>::const_iterator Project::findPattern(const std::string &instance_id) const {
    auto result = std::find_if(patterns_.cbegin(), patterns_.cend(), [&instance_id](const auto &value) {
        return value->usesInstanceId(instance_id);
    });
    return result;
}
} // traact