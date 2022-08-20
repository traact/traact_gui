/** Copyright (C) 2022  Frieder Pankratz <frieder.pankratz@gmail.com> **/

#include "WindowDataflow.h"
#include "external/ImFileDialog/ImFileDialog.h"

namespace traact::gui::window {
WindowDataflow::WindowDataflow(state::ApplicationState &state) : Window("Graph", state) {}
void WindowDataflow::render() {

    for (const auto &dataflow : state_.open_files) {

        if (!dataflow->open) {
            continue;
        }

        int flags{ImGuiWindowFlags_None};
        if (dataflow->dirty) {
            flags |= ImGuiWindowFlags_UnsavedDocument ;
        }
        ImGui::Begin(dataflow->getName(), nullptr, flags);


        if(ImGui::IsWindowFocused(ImGuiFocusedFlags_RootAndChildWindows)){
            if(!state_.selected_traact_element.isCurrentDataflow(dataflow)){
                state_.selected_traact_element.setSelected(dataflow);
            }

        } else {
            if(state_.selectionChangedTo(dataflow)){
                ImGui::SetWindowFocus();
            }
        }

        dataflow->drawContextMenu();
        dataflow->draw();

        ImGui::End();
    }
}
} // traact