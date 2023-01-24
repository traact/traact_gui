/** Copyright (C) 2022  Frieder Pankratz <frieder.pankratz@gmail.com> **/

#ifndef TRAACT_GUI_SRC_TRAACT_STATE_PROJECT_PATTERN_PATTERN_H_
#define TRAACT_GUI_SRC_TRAACT_STATE_PROJECT_PATTERN_PATTERN_H_

#include <string>

namespace traact::gui::state::project {

class Pattern {
 public:
    Pattern() = default;
    virtual ~Pattern() = default;

    virtual bool usesInstanceId(const std::string& instance_id) = 0;
};

} // traact

#endif //TRAACT_GUI_SRC_TRAACT_STATE_PROJECT_PATTERN_PATTERN_H_
