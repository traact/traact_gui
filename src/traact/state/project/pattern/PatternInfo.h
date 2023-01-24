/** Copyright (C) 2022  Frieder Pankratz <frieder.pankratz@gmail.com> **/

#ifndef TRAACT_GUI_SRC_TRAACT_STATE_PROJECT_PATTERN_PATTERNINFO_H_
#define TRAACT_GUI_SRC_TRAACT_STATE_PROJECT_PATTERN_PATTERNINFO_H_

#include <string>
#include <vector>

namespace traact::gui::state::project  {

struct PatternInfo {
    PatternInfo(std::string pattern_id,
                std::string display_name,
                std::string description,
                std::vector<std::string> tags);
    PatternInfo();
    std::string pattern_id;
    std::string display_name;
    std::string description;
    std::vector<std::string> tags;

    bool containsTag(const std::string& tag);

};

} // traact

#endif //TRAACT_GUI_SRC_TRAACT_STATE_PROJECT_PATTERN_PATTERNINFO_H_
