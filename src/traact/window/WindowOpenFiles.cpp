/** Copyright (C) 2022  Frieder Pankratz <frieder.pankratz@gmail.com> **/

#include "WindowOpenFiles.h"

namespace traact::gui::window {

WindowOpenFiles::WindowOpenFiles(state::ApplicationState &state) : Window("OpenFiles", state) {}
void WindowOpenFiles::render() {
    static constexpr ImGuiTreeNodeFlags base_flags = ImGuiTreeNodeFlags_OpenOnArrow | ImGuiTreeNodeFlags_SpanAvailWidth;

    ImGui::Begin("Open files");

    ImGui::SetNextItemOpen(true);
    if (ImGui::TreeNode("Loaded Dataflow")) {

        for (const auto &dataflow : state_.open_files) {
            if (!dataflow->open)
                continue;

            ImGuiTreeNodeFlags node_flags = base_flags;
            if (state_.selected_traact_element.isCurrentDataflow(dataflow)) {
                node_flags |= ImGuiTreeNodeFlags_Selected;
            }

            bool node_open = ImGui::TreeNodeEx(dataflow.get(), node_flags, "%s", dataflow->getName());
            if (ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left) && ImGui::IsItemHovered()) {
                state_.selected_traact_element.setSelected(dataflow);

            }

            if (node_open) {

                auto all_patterns = dataflow->graph_editor_.Graph->getAll();
                if (ImGui::BeginListBox("##pattern_instances",
                                        ImVec2(-FLT_MIN,
                                               all_patterns.size() * ImGui::GetTextLineHeightWithSpacing() + 2))) {
                    for (const auto &tmp : all_patterns) {
                        bool is_selected = state_.selected_traact_element.isSelected(tmp);

                        if (ImGui::Selectable(tmp->instance_id.c_str(), is_selected)) {
                            state_.selected_traact_element.setSelected(tmp);
                        }
                        if (is_selected) {
                            ImGui::SetItemDefaultFocus();
                        }
                    }
                    ImGui::EndListBox();
                }

                ImGui::TreePop();
            }
        }

        ImGui::TreePop();
    }


    //ImGui::SetNextTreeNodeOpen(true);
    if (ImGui::TreeNode("All Patterns")) {
        int i = 0;
        for (const auto &tmp : state_.available_patterns) {

            //ImGui::PushID(i);

            ImGui::Button(tmp->name.c_str());
            if (ImGui::BeginDragDropSource(ImGuiDragDropFlags_None)) {

                ImGui::SetDragDropPayload("NEW_PATTERN_DRAGDROP", tmp->name.c_str(), tmp->name.length() + 1);

                ImGui::EndDragDropSource();
            }


            //ImGui::PopID();
            i++;

        }

        ImGui::TreePop();
    }

    ImGui::End();

}
}