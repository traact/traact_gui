/** Copyright (C) 2022  Frieder Pankratz <frieder.pankratz@gmail.com> **/

#ifndef TRAACT_GUI_SRC_TRAACT_EDITORELEMENT_H_
#define TRAACT_GUI_SRC_TRAACT_EDITORELEMENT_H_

#include <memory>
#include "traact/state/ApplicationState.h"

namespace traact::gui {
class Window {
 public:
    using SharedPtr = std::shared_ptr<Window>;
    Window(std::string name, state::ApplicationState &state);
    virtual ~Window() = default;

    const std::string& name() const;


    virtual void init();
    virtual void render();
    /**
     *
     * @return true if the updateMovement loop should keep running
     */
    virtual bool renderStop();
    virtual void destroy();
 protected:
    std::string name_;
    state::ApplicationState& state_;


};
}

#endif //TRAACT_GUI_SRC_TRAACT_EDITORELEMENT_H_
