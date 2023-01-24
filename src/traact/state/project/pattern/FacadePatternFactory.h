/** Copyright (C) 2022  Frieder Pankratz <frieder.pankratz@gmail.com> **/

#ifndef TRAACT_GUI_SRC_TRAACT_STATE_PROJECT_PATTERN_FACADEPATTERNFACTORY_H_
#define TRAACT_GUI_SRC_TRAACT_STATE_PROJECT_PATTERN_FACADEPATTERNFACTORY_H_

#include "PatternFactory.h"

namespace traact::gui::state::project {

class FacadePatternFactory : public project::PatternFactory{
 public:
    FacadePatternFactory();
    const std::vector<PatternInfo> &getAllPatterns() override;
    std::shared_ptr<Pattern> createPattern(const std::string &pattern_id,
                                                   const std::string &instance_id) override;

 private:
    std::vector<PatternInfo> all_patterns_;

};

} // traact

#endif //TRAACT_GUI_SRC_TRAACT_STATE_PROJECT_PATTERN_FACADEPATTERNFACTORY_H_
