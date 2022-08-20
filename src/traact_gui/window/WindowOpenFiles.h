/** Copyright (C) 2022  Frieder Pankratz <frieder.pankratz@gmail.com> **/

#ifndef TRAACT_GUI_SRC_TRAACT_GUI_WINDOW_WINDOWOPENFILES_H_
#define TRAACT_GUI_SRC_TRAACT_GUI_WINDOW_WINDOWOPENFILES_H_

#include "traact_gui/app/Window.h"

namespace traact::gui::window {
class WindowOpenFiles : public Window{
 public:
    WindowOpenFiles(state::ApplicationState &state);
    virtual void render() override;
};
}


#endif //TRAACT_GUI_SRC_TRAACT_GUI_WINDOW_WINDOWOPENFILES_H_
