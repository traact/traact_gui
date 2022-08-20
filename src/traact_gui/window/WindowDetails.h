/** Copyright (C) 2022  Frieder Pankratz <frieder.pankratz@gmail.com> **/

#ifndef TRAACT_GUI_SRC_TRAACT_GUI_WINDOW_WINDOWDETAILS_H_
#define TRAACT_GUI_SRC_TRAACT_GUI_WINDOW_WINDOWDETAILS_H_

#include "traact_gui/app/Window.h"
#include "traact_gui/editor/DetailsEditor.h"

namespace traact::gui::window {
class WindowDetails : public Window {
 public:
    WindowDetails(state::ApplicationState &state);
    virtual void render() override;

 private:
    DetailsEditor details_editor_;
};
}


#endif //TRAACT_GUI_SRC_TRAACT_GUI_WINDOW_WINDOWDETAILS_H_
