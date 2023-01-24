/** Copyright (C) 2022  Frieder Pankratz <frieder.pankratz@gmail.com> **/

#ifndef TRAACT_GUI_SRC_TRAACT_STATE_PROJECT_PROJECT_H_
#define TRAACT_GUI_SRC_TRAACT_STATE_PROJECT_PROJECT_H_

#include "pattern/PatternFactory.h"
#include <vector>
#include <memory>
#include <optional>
namespace traact::gui::state::project {


class Project {
 public:
    explicit Project(const std::shared_ptr<PatternFactory> &pattern_factory);
    ~Project() = default;


    std::shared_ptr<Pattern> addPattern(const std::string& pattern_id);
    std::shared_ptr<Pattern> addPattern(const std::string& pattern_id, const std::string& instance_id);


 private:
    std::shared_ptr<PatternFactory> pattern_factory_;
    std::vector<std::shared_ptr<Pattern>> patterns_;

    [[nodiscard]] std::vector<std::shared_ptr<Pattern>>::const_iterator findPattern(const std::string &instance_id) const;
};

} // traact

#endif //TRAACT_GUI_SRC_TRAACT_STATE_PROJECT_PROJECT_H_
