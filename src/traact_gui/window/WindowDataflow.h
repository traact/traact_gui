/** Copyright (C) 2022  Frieder Pankratz <frieder.pankratz@gmail.com> **/

#ifndef TRAACT_GUI_SRC_TRAACT_GUI_WINDOW_WINDOWDATAFLOW_H_
#define TRAACT_GUI_SRC_TRAACT_GUI_WINDOW_WINDOWDATAFLOW_H_

#include "traact_gui/app/Window.h"

namespace traact::gui::window {

class WindowDataflow : public Window{
 public:
    WindowDataflow(state::ApplicationState &state);
    void render() override;

};

} // traact

#endif //TRAACT_GUI_SRC_TRAACT_GUI_WINDOW_WINDOWDATAFLOW_H_
