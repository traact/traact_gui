/** Copyright (C) 2022  Frieder Pankratz <frieder.pankratz@gmail.com> **/

#include "FacadePatternFactory.h"
#include <traact/traact.h>
namespace traact::gui::state::project {

FacadePatternFactory::FacadePatternFactory() {
    DefaultFacade default_facade;
    for(const auto& pattern : default_facade.GetAllAvailablePatterns()) {
        all_patterns_.emplace_back(pattern->name, pattern->display_name, pattern->description, pattern->tags);
    };
}
const std::vector<PatternInfo> &FacadePatternFactory::getAllPatterns() {
    return all_patterns_;
}
std::shared_ptr<Pattern> FacadePatternFactory::createPattern(const std::string &pattern_id,
                                                            const std::string &instance_id) {
    return std::shared_ptr<Pattern>();
}
} // traact