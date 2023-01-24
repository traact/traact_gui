/** Copyright (C) 2022  Frieder Pankratz <frieder.pankratz@gmail.com> **/

#ifndef TRAACT_GUI_SRC_TRAACT_WINDOW_OPENPROJECTS_H_
#define TRAACT_GUI_SRC_TRAACT_WINDOW_OPENPROJECTS_H_

#include "traact/app/Window.h"

namespace traact::gui::window  {

class OpenProjects : public Window {
 public:
    OpenProjects(const std::string &name, state::ApplicationState &state);
    virtual void render() override;
};

} // traact

#endif //TRAACT_GUI_SRC_TRAACT_WINDOW_OPENPROJECTS_H_
