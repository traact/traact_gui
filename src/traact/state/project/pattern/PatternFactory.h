/** Copyright (C) 2022  Frieder Pankratz <frieder.pankratz@gmail.com> **/

#ifndef TRAACT_GUI_SRC_TRAACT_STATE_PROJECT_PATTERN_PATTERNFACTORY_H_
#define TRAACT_GUI_SRC_TRAACT_STATE_PROJECT_PATTERN_PATTERNFACTORY_H_

#include "Pattern.h"
#include "PatternInfo.h"
#include <memory>
namespace traact::gui::state::project {
class PatternFactory {
 public:
    PatternFactory() = default;
    virtual ~PatternFactory() = default;

    virtual const std::vector<PatternInfo> & getAllPatterns() = 0;
    virtual std::shared_ptr<Pattern> createPattern(const std::string& pattern_id, const std::string& instance_id) = 0;
};
}

#endif //TRAACT_GUI_SRC_TRAACT_STATE_PROJECT_PATTERN_PATTERNFACTORY_H_
