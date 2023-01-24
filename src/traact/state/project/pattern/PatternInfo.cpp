/** Copyright (C) 2022  Frieder Pankratz <frieder.pankratz@gmail.com> **/

#include "PatternInfo.h"

#include <utility>
#include <algorithm>
namespace traact::gui::state::project  {
PatternInfo::PatternInfo(std::string pattern_id,
                         std::string display_name,
                         std::string description,
                         std::vector<std::string> tags)
    : pattern_id(std::move(pattern_id)), display_name(std::move(display_name)), description(std::move(description)), tags(std::move(tags)) {}
PatternInfo::PatternInfo() : pattern_id{"invalid"}, display_name{"invalid"}, description{}, tags{} {}
bool PatternInfo::containsTag(const std::string &tag) {
    auto contains_tag = std::find(tags.begin(),tags.end(), tag);
    return contains_tag != tags.end();
}

} // traact