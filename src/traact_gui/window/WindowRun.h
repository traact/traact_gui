/** Copyright (C) 2022  Frieder Pankratz <frieder.pankratz@gmail.com> **/

#ifndef TRAACT_GUI_SRC_TRAACT_GUI_WINDOW_WINDOWRUN_H_
#define TRAACT_GUI_SRC_TRAACT_GUI_WINDOW_WINDOWRUN_H_

#include "traact_gui/app/Window.h"
#include "traact_gui/debug_run/DebugRun.h"
namespace traact::gui::window {

class WindowRun : public  Window{
 public:
    WindowRun(state::ApplicationState &state);
    void render() override;
 private:
    DebugRun debug_run_;
};

} // traact

#endif //TRAACT_GUI_SRC_TRAACT_GUI_WINDOW_WINDOWRUN_H_
