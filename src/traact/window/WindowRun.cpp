/** Copyright (C) 2022  Frieder Pankratz <frieder.pankratz@gmail.com> **/


#include "WindowRun.h"

namespace traact::gui::window  {
WindowRun::WindowRun(state::ApplicationState &state) : Window("Run", state) {}
void WindowRun::render() {
    if(state_.selectionChanged()){
        debug_run_.setCurrentDataflow(state_.selected_traact_element.current_dataflow);


    }
    debug_run_.draw();
}
} // traact