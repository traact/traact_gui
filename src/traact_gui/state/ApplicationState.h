/** Copyright (C) 2022  Frieder Pankratz <frieder.pankratz@gmail.com> **/

#ifndef TRAACT_GUI_SRC_TRAACT_GUI_APPLICATIONSTATE_H_
#define TRAACT_GUI_SRC_TRAACT_GUI_APPLICATIONSTATE_H_

#include <nlohmann/json.hpp>

namespace traact::gui::state {

struct ApplicationState {
    nlohmann::json config;
};

} // traact

#endif //TRAACT_GUI_SRC_TRAACT_GUI_APPLICATIONSTATE_H_
