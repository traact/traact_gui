/** Copyright (C) 2022  Frieder Pankratz <frieder.pankratz@gmail.com> **/

#ifndef TRAACT_GUI_SRC_TRAACT_WINDOW_MENUFILE_H_
#define TRAACT_GUI_SRC_TRAACT_WINDOW_MENUFILE_H_

#include "traact/app/Window.h"
#include "traact/state/OpenFiles.h"

namespace traact::gui::window {

class MenuFile : public Window {
 public:
    MenuFile(state::ApplicationState &state);
    virtual void render() override;

};

} // traact


#endif //TRAACT_GUI_SRC_TRAACT_WINDOW_MENUFILE_H_
