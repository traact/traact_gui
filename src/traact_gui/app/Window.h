/** Copyright (C) 2022  Frieder Pankratz <frieder.pankratz@gmail.com> **/

#ifndef TRAACT_GUI_SRC_TRAACT_GUI_EDITORELEMENT_H_
#define TRAACT_GUI_SRC_TRAACT_GUI_EDITORELEMENT_H_

#include <memory>
#include "traact_gui/state/ApplicationState.h"

namespace traact::gui {
class Window {
 public:
    Window(state::ApplicationState &state);
    virtual ~Window() = default;

    virtual void onInit();
    virtual void onRender();
    virtual void onDestroy();
 protected:
    state::ApplicationState& state_;

};

using WindowPtr = std::shared_ptr<Window>;
}

#endif //TRAACT_GUI_SRC_TRAACT_GUI_EDITORELEMENT_H_
