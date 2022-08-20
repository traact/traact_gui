/** Copyright (C) 2022  Frieder Pankratz <frieder.pankratz@gmail.com> **/

#include <imgui.h>
#define IMGUI_DEFINE_MATH_OPERATORS
#include <imgui_internal.h>
#include "MenuFile.h"
#include "external/ImFileDialog/ImFileDialog.h"

namespace traact::gui::window {

MenuFile::MenuFile(state::ApplicationState &state) : Window("FilesMenu", state) {}
void window::MenuFile::render() {
    auto &gui_io = ImGui::GetIO();
    auto current_dataflow = state_.selected_traact_element.current_dataflow;

    if (ImGui::BeginMainMenuBar()) {
        if (ImGui::BeginMenu("File")) {

            if (ImGui::MenuItem("New")) {
                state_.newFile();
            }
            if (ImGui::MenuItem("Open", "Ctrl+O")) {
                ifd::FileDialog::Instance().Open("DataflowOpenDialog",
                                                 "Open a dataflow",
                                                 "Json file (*.json){.json},.*",
                                                 false);
            }
            if (ImGui::BeginMenu("Open Recent")) {
                for (const auto &file_name : state_.recent_files) {
                    if (ImGui::MenuItem(file_name.c_str())) {
                        state_.openFile(file_name);
                    }
                }
                ImGui::EndMenu();
            }

            if (ImGui::MenuItem("Save", "Ctrl+S", false, state_.selected_traact_element.current_dataflow != nullptr)) {
                if (current_dataflow->dirty)
                    current_dataflow->doQueueSave();
            }
            if (ImGui::MenuItem("Save all")) {
                for (auto &dataflow : state_.open_files)
                    if (dataflow && dataflow->dirty)
                        dataflow->doQueueSave();
            }
            if (ImGui::MenuItem("Save As..", nullptr, false, current_dataflow != nullptr)) {
                state_.newFile(current_dataflow->toDataString());
            }
            if (ImGui::MenuItem("Close All", nullptr, false, !state_.open_files.empty())) {
                for (auto &dataflow : state_.open_files)
                    dataflow->doQueueClose();
            }

            ImGui::Separator();

            if (ImGui::MenuItem("Quit", "Alt+F4")) {}

            ImGui::EndMenu();
        }
        if (ImGui::BeginMenu("Edit")) {
            if (ImGui::MenuItem("undo", "CTRL+Z", false, current_dataflow->canUndo())) {
                current_dataflow->undo();
            }
            if (ImGui::MenuItem("redo", "CTRL+Y", false, current_dataflow->canRedo())) {
                current_dataflow->redo();
            }
            ImGui::EndMenu();
        }

        ImGui::EndMainMenuBar();
    }

    if (ifd::FileDialog::Instance().IsDone("DataflowOpenDialog")) {
        if (ifd::FileDialog::Instance().HasResult()) {
            const fs::path &res = ifd::FileDialog::Instance().GetResult();
            spdlog::info("open file: {0}", res.string());
            state_.openFile(res.string());

        }
        ifd::FileDialog::Instance().Close();
    }
}
} // traact