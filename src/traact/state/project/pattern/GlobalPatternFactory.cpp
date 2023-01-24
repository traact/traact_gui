/** Copyright (C) 2022  Frieder Pankratz <frieder.pankratz@gmail.com> **/

#include "GlobalPatternFactory.h"
#include <spdlog/spdlog.h>
namespace traact::gui::state::project {
std::shared_ptr<Pattern> GlobalPatternFactory::createPattern(const std::string &pattern_id,
                                                             const std::string &instance_id) {
    auto factory = pattern_id_to_factory_.find(pattern_id);
    if(factory == pattern_id_to_factory_.cend()){
        SPDLOG_ERROR("pattern id not found: {0}", pattern_id);
        return {};
    } else {
        return factory->second->createPattern(pattern_id, instance_id);
    }
}
const std::vector<PatternInfo> & GlobalPatternFactory::getAllPatterns() {
    return all_patterns_;
}
void GlobalPatternFactory::addFactory(std::shared_ptr<PatternFactory> factory) {
    for(const auto& pattern_info : factory->getAllPatterns()) {
        auto pattern_exists = pattern_id_to_factory_.find(pattern_info.pattern_id);
        if(pattern_exists == pattern_id_to_factory_.end()){
            all_patterns_.push_back(pattern_info);
            pattern_id_to_factory_.emplace(pattern_info.pattern_id, factory);
        } else {
            SPDLOG_DEBUG("pattern with id: \"{0}\" already loaded, skipping", pattern_info.pattern_id);
        }
    }

}
} // traact