/** Copyright (C) 2022  Frieder Pankratz <frieder.pankratz@gmail.com> **/

#ifndef TRAACT_GUI_SRC_TRAACT_STATE_PROJECT_PATTERN_GLOBALPATTERNFACTORY_H_
#define TRAACT_GUI_SRC_TRAACT_STATE_PROJECT_PATTERN_GLOBALPATTERNFACTORY_H_

#include "PatternFactory.h"
#include <unordered_map>
#include <memory>

namespace traact::gui::state::project {

class GlobalPatternFactory : public PatternFactory{

 public:
    const std::vector<PatternInfo> & getAllPatterns() override;
    std::shared_ptr<Pattern> createPattern(const std::string &pattern_id,
                                                   const std::string &instance_id) override;
    void addFactory(std::shared_ptr<PatternFactory> factory);
 private:
    std::unordered_map<std::string, std::shared_ptr<PatternFactory>> pattern_id_to_factory_;
    std::vector<PatternInfo> all_patterns_;
};

} // traact

#endif //TRAACT_GUI_SRC_TRAACT_STATE_PROJECT_PATTERN_GLOBALPATTERNFACTORY_H_
