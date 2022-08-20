/** Copyright (C) 2022  Frieder Pankratz <frieder.pankratz@gmail.com> **/

#include "WindowDetails.h"

namespace traact::gui::window {

WindowDetails::WindowDetails(state::ApplicationState &state) : Window("Details", state) {

    details_editor_.onChange = [this](const TraactElement& element){
        state_.dataflowDetailsChanged(element);
    };

}
void WindowDetails::render() {
    ImGui::Begin("Details");
    std::visit(details_editor_, state_.selected_traact_element.selected);
    ImGui::End();
}
}