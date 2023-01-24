/** Copyright (C) 2022  Frieder Pankratz <frieder.pankratz@gmail.com> **/

#include "ApplicationState.h"
#include "external/ImFileDialog/ImFileDialog.h"
#include <fstream>

#include "project/pattern/FacadePatternFactory.h"
namespace traact::gui::state {


ApplicationState::ApplicationState(const std::string &config_file) : config_file(config_file) {

    auto& new_project = open_projects.emplace_back();

    pattern_factory = std::make_shared<project::GlobalPatternFactory>();
    pattern_factory->addFactory(std::make_shared<project::FacadePatternFactory>());

}

void ApplicationState::saveConfig() {
    try {
        nlohmann::json json_graph;

        for (const auto &file : recent_files) {
            json_graph["RecentFiles"].push_back(file);
        }

        std::ofstream config_stream;
        config_stream.open(config_file);
        config_stream << json_graph.dump(4);
        config_stream.close();
    } catch (...) {
        spdlog::error("error saving config file");
    }

}

void ApplicationState::loadConfig() {
    try {
        if (util::fileExists(config_file)) {
            nlohmann::json json_graph;
            std::ifstream config_stream;
            config_stream.open(config_file);
            config_stream >> json_graph;
            config_stream.close();

            auto recent_result = json_graph.find("RecentFiles");

            if (recent_result != json_graph.end()) {
                for (auto &recent_file : *recent_result) {
                    recent_files.push_back(recent_file.get<std::string>());
                }
            }
        } else {
            SPDLOG_INFO("no config file");
        }

    } catch (...) {
        spdlog::error("error loading config file");
    }

}

void ApplicationState::openFile(fs::path file) {
    if (!exists(file)) {
        spdlog::error("file {0} does not exist", file.string());
        return;
    }

    auto *find_result = std::find_if(recent_files.begin(),
                                     recent_files.end(),
                                     [&file](const std::string &x) { return x == file.string(); });
    if (find_result == recent_files.end()) {
        recent_files.push_back(file.string());
        saveConfig();
    }

    try {
        open_files.push_back(std::make_shared<DataflowFile>(file, selected_traact_element));
        //auto& new_project = open_projects.emplace_back<std::shared_ptr<project::Project>>();
    } catch (...) {
        spdlog::error("error loading file {0}", file.string());
    }

}

const std::vector<std::optional<std::string>> &ApplicationState::openFiles() {
    return open_file_paths;
}

void ApplicationState::closeFile(std::string file) {

}

void ApplicationState::closeAll() {

}

void ApplicationState::newFile() {
    for (int i = 0; i < 256; ++i) {
        std::string new_name = fmt::format("untitled {0}", i);
        auto result = std::find_if(
            open_files.begin(), open_files.end(),
            [&new_name](const auto &x) { return x->getNameString() == new_name; });
        if (result == open_files.end()) {
            open_files.push_back(std::make_shared<DataflowFile>(new_name, selected_traact_element));
            return;
        }
    }

    spdlog::error("too many untitled dataflow networks open. unable to create new dataflow");
}
void ApplicationState::newFile(const std::string &dataflow_json) {
    nlohmann::json json_graph;
    std::stringstream ss(dataflow_json);
    ss >> json_graph;
    auto new_dataflow = std::make_shared<DataflowFile>(json_graph, selected_traact_element);
    new_dataflow->doQueueSave();
    open_files.push_back(new_dataflow);
}
void ApplicationState::update() {
    for (auto &dataflow : open_files) {

        if (!dataflow->open) {
            continue;
        }

        // Cancel attempt to close when unsaved add to save queue so we can display a popup.
        if (!dataflow->open && dataflow->dirty) {
            dataflow->open = true;
            dataflow->doQueueClose();
        }

    }


    // Update closing queue
    if (close_queue.empty()) {
        // Close queue is locked once we started a popup
        for (int doc_n = 0; doc_n < open_files.size(); doc_n++) {
            auto &dataflow = open_files[doc_n];
            if (dataflow->wantClose) {
                dataflow->wantClose = false;
                close_queue.push_back(dataflow);
            }
        }
    }

    // Display closing confirmation UI
    if (!close_queue.empty()) {
        int close_queue_unsaved_documents = 0;
        for (int n = 0; n < close_queue.size(); n++)
            if (close_queue[n]->dirty)
                close_queue_unsaved_documents++;

        if (close_queue_unsaved_documents == 0) {
            // Close documents when all are clean
            for (int n = 0; n < close_queue.size(); n++)
                close_queue[n]->doForceClose();
            close_queue.clear();
        } else {
            if (!ImGui::IsPopupOpen("Save?"))
                ImGui::OpenPopup("Save?");
            if (ImGui::BeginPopupModal("Save?", NULL, ImGuiWindowFlags_AlwaysAutoResize)) {
                ImGui::Text("Save change to the following items?");
                float item_height = ImGui::GetTextLineHeightWithSpacing();
                if (ImGui::BeginChildFrame(ImGui::GetID("frame"), ImVec2(-FLT_MIN, 6.25f * item_height))) {
                    for (int n = 0; n < close_queue.size(); n++)
                        if (close_queue[n]->dirty)
                            ImGui::Text("%s", close_queue[n]->getName());
                    ImGui::EndChildFrame();
                }

                ImVec2 button_size(ImGui::GetFontSize() * 7.0f, 0.0f);
                if (ImGui::Button("Yes", button_size)) {
                    for (int n = 0; n < close_queue.size(); n++) {
                        if (close_queue[n]->dirty) {
                            close_queue[n]->doQueueSave();
                        }
                        close_queue[n]->doForceClose();
                    }
                    close_queue.clear();
                    ImGui::CloseCurrentPopup();
                }
                ImGui::SameLine();
                if (ImGui::Button("No", button_size)) {
                    for (int n = 0; n < close_queue.size(); n++)
                        close_queue[n]->doForceClose();
                    close_queue.clear();
                    ImGui::CloseCurrentPopup();
                }
                ImGui::SameLine();
                if (ImGui::Button("Cancel", button_size)) {
                    close_queue.clear();
                    ImGui::CloseCurrentPopup();
                }
                ImGui::EndPopup();
            }
        }

    }

    if (save_queue.empty()) {
        // Save queue is locked once we started a popup
        for (int doc_n = 0; doc_n < open_files.size(); doc_n++) {
            auto &dataflow = open_files[doc_n];
            if (dataflow->wantSave) {
                dataflow->wantSave = false;
                save_queue.push(dataflow);
            }
        }
    } else {
        static bool show_save_dialog = false;

        auto &current = save_queue.top();
        if (!show_save_dialog) {
            if (!current->filepath.has_filename()) {
                ifd::FileDialog::Instance().Save("##DataflowSaveDialog",
                                                 fmt::format("Save {0}", current->getNameString()),
                                                 "*.json {.json}");
                show_save_dialog = true;
            } else {
                current->doSave();
                save_queue.pop();
                show_save_dialog = false;
            }

        }

        if (ifd::FileDialog::Instance().IsDone("##DataflowSaveDialog")) {

            if (ifd::FileDialog::Instance().HasResult()) {
                current->filepath = ifd::FileDialog::Instance().GetResult();
                spdlog::info("SAVE {1} {0}", current->filepath.string(), current->getNameString());
                current->doSave();
            }
            save_queue.pop();
            show_save_dialog = false;
            ifd::FileDialog::Instance().Close();
        }
    }
    last_selected_dataflow = selected_traact_element.current_dataflow;
}
bool ApplicationState::selectionChanged() {

    return !selected_traact_element.isCurrentDataflow(last_selected_dataflow);
}
void ApplicationState::dataflowDetailsChanged(const TraactElement &changed_element) {
    std::visit(overloaded {
        [this](auto element) {
            selected_traact_element.current_dataflow->saveState();
        },
        [](std::shared_ptr<DataflowFile> element){

            element->saveState();
        }
    }, changed_element);
}
bool ApplicationState::selectionChangedTo(const std::shared_ptr<DataflowFile> &dataflow) {
    return selectionChanged() && selected_traact_element.isCurrentDataflow(dataflow);
}
const std::vector<project::PatternInfo>& ApplicationState::getAvailablePatterns() const{
    return pattern_factory->getAllPatterns();
}

} // traact