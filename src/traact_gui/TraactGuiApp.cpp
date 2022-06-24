/*  BSD 3-Clause License
 *
 *  Copyright (c) 2020, FriederPankratz <frieder.pankratz@gmail.com>
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions are met:
 *
 *  1. Redistributions of source code must retain the above copyright notice, this
 *     list of conditions and the following disclaimer.
 *
 *  2. Redistributions in binary form must reproduce the above copyright notice,
 *     this list of conditions and the following disclaimer in the documentation
 *     and/or other materials provided with the distribution.
 *
 *  3. Neither the name of the copyright holder nor the names of its
 *     contributors may be used to endorse or promote products derived from
 *     this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 *  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 *  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 *  DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 *  FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 *  DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 *  SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 *  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 *  OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 *  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
**/

#include "TraactGuiApp.h"

#include <traact/facade/DefaultFacade.h>
#include <traact/serialization/JsonGraphInstance.h>
#include <fstream>

#include "ImGuiUtils.h"
#include <implot.h>
#include <external/ImFileDialog/ImFileDialog.h>
#include <external/imgui-node-editor/imgui_node_editor.h>

traact::gui::TraactGuiApp::TraactGuiApp(const std::string &configFile) : config_file_(configFile) {
    facade::DefaultFacade facade;


    available_patterns_ = facade.GetAllAvailablePatterns();

    LoadConfig();

}

traact::gui::TraactGuiApp::~TraactGuiApp() {
    // TODO check for unsaved files?
    for (auto flow : dataflow_files_) {
        delete flow;
    }
}

void traact::gui::TraactGuiApp::OpenFile(fs::path file) {
    if(!exists(file)){
        spdlog::error("file {0} does not exist", file.string());
        return;
    }

    auto find_result = std::find_if(recent_files_.begin(), recent_files_.end(),  [&file](const std::string& x) { return x == file.string();});
    if(find_result == recent_files_.end()){
        recent_files_.push_back(file.string());
        SaveConfig();
    }




    try{
        DataflowFile* new_dataflow = new DataflowFile(file);
        dataflow_files_.push_back(new_dataflow);
    } catch (...) {
        spdlog::error("error loading file {0}", file.string());
    }



}

const std::vector<std::string> & traact::gui::TraactGuiApp::OpenFiles() {
    return open_files_;
}

void traact::gui::TraactGuiApp::CloseFile(std::string file) {


}

void traact::gui::TraactGuiApp::CloseAll() {

}




void traact::gui::TraactGuiApp::NewFile() {
    for (int i = 0; i < 256; ++i) {
        std::string new_name = fmt::format("untitled {0}", i);
        auto result = std::find_if(
                dataflow_files_.begin(), dataflow_files_.end(),
                [&new_name](const DataflowFile* x) { return x->GetNameString() == new_name;});
        if(result == dataflow_files_.end()){
            dataflow_files_.push_back(new DataflowFile(new_name));
            return;
        }
    }

    spdlog::error("too many untitled dataflow networks open. unable to create new dataflow");
}

void traact::gui::TraactGuiApp::OnFrame() {

    MenuBar();

    if(current_dataflow_ == nullptr && !dataflow_files_.empty()){
        pending_dataflow_ = dataflow_files_[0];
    }

    static float leftPaneWidth  = 400.0f;
    static float rightPaneWidth = 800.0f;
    Splitter("##Splitter",true, 4.0f, &leftPaneWidth, &rightPaneWidth, 50.0f, 50.0f);


    DrawLeftPanel(leftPaneWidth, -1);

    ImGui::SameLine(0.0f, 12.0f);

    DrawRightPanel(rightPaneWidth, -1);







}

void traact::gui::TraactGuiApp::MenuBar() {
    auto& io = ImGui::GetIO();

    static bool show_imgui_demo = false;
    if (ImGui::BeginMainMenuBar())
    {
        if (ImGui::BeginMenu("File"))
        {
            if (ImGui::MenuItem("New")) {
                NewFile();
            }
            if (ImGui::MenuItem("Open", "Ctrl+O"))
            {
                ifd::FileDialog::Instance().Open("DataflowOpenDialog", "Open a dataflow", "Json file (*.json){.json},.*", false);
            }
            if (ImGui::BeginMenu("Open Recent"))
            {
                for(const auto& file_name : recent_files_){
                    if(ImGui::MenuItem(file_name.c_str())){
                        OpenFile(file_name);
                    }
                }
                ImGui::EndMenu();
            }
            if (ImGui::MenuItem("Save", "Ctrl+S", false, current_dataflow_ != nullptr)) {
                if(current_dataflow_->dirty)
                    current_dataflow_->DoQueueSave();
            }
            if (ImGui::MenuItem("Save all")) {
                for (auto& dataflow : dataflow_files_)
                    if(dataflow && dataflow->dirty)
                        dataflow->DoQueueSave();
            }
            if (ImGui::MenuItem("Save As..", nullptr , false, current_dataflow_ != nullptr)) {
                NewFile(current_dataflow_->ToDataString());
            }
            if (ImGui::MenuItem("Close All", nullptr, false, !dataflow_files_.empty())){
                for (auto& dataflow : dataflow_files_)
                    dataflow->DoQueueClose();
            }


            ImGui::Separator();

            if (ImGui::MenuItem("Quit", "Alt+F4")) {}

            ImGui::EndMenu();
        }
        if (ImGui::BeginMenu("Edit"))
        {
            if (ImGui::MenuItem("Undo", "CTRL+Z", false, current_dataflow_->CanUndo()))
                current_dataflow_->Undo();

            if (ImGui::MenuItem("Redo", "CTRL+Y", false, current_dataflow_->CanRedo()))
                current_dataflow_->Redo();

//            ImGui::Separator();
//            if (ImGui::MenuItem("Cut", "CTRL+X")) {}
//            if (ImGui::MenuItem("Copy", "CTRL+C")) {}
//            if (ImGui::MenuItem("Paste", "CTRL+V")) {}
            ImGui::EndMenu();
        }
        if(ImGui::BeginMenu("Help")) {
            if(ImGui::MenuItem("ImGui Demo Window")){
                show_imgui_demo = !show_imgui_demo;
            }
            ImGui::EndMenu();
        }
        ImGui::EndMainMenuBar();
    }

    if (ifd::FileDialog::Instance().IsDone("DataflowOpenDialog")) {
        if (ifd::FileDialog::Instance().HasResult()) {
            const fs::path& res = ifd::FileDialog::Instance().GetResult();
            spdlog::info("open file: {0}", res.string());
            OpenFile(res.string());

        }
        ifd::FileDialog::Instance().Close();
    }

    if(show_imgui_demo){
        ImGui::ShowDemoWindow();
    }
}

void traact::gui::TraactGuiApp::DrawLeftPanel(int width, int height) {
    static float patternHeight  = 600;
    static float detailHeight = 50.0f;

    Splitter("##SplitterLeft",false, 4.0f, &patternHeight, &detailHeight, 50.0f, 50.0f,width);
    ImGui::BeginChild("##leftMainPanel", ImVec2(width, patternHeight+detailHeight));

    //ImGuiWindowFlags_AlwaysVerticalScrollbar | ImGuiWindowFlags_HorizontalScrollbar
    ImGui::BeginChild("##mainPatternPanel", ImVec2(width, patternHeight+4));
    DrawPatternPanel();
    ImGui::EndChild();

    ImGui::BeginChild("##mainDetailsPanel");
    DrawDetailsPanel();
    ImGui::EndChild();


    ImGui::EndChild();
}

void traact::gui::TraactGuiApp::DrawRightPanel(int width, int height) {

    // Options
    static ImGuiTabBarFlags tab_bar_flags = ImGuiTabBarFlags_FittingPolicyDefault_ |  ImGuiTabBarFlags_Reorderable | ImGuiTabBarFlags_AutoSelectNewTabs;

    ImGui::BeginChild("##rightMainPanel");
    // Submit Tab Bar and Tabs
    {
        if (ImGui::BeginTabBar("##dataflow_tabs", tab_bar_flags))
        {
            for (const auto& dataflow : dataflow_files_) {
                if (!dataflow->open && dataflow->openPrev)
                    ImGui::SetTabItemClosed(dataflow->GetName());
                dataflow->openPrev = dataflow->open;
            }

            // Submit Tabs
            for (const auto& dataflow : dataflow_files_) {

                if (!dataflow->open)
                    continue;

                ImGuiTabItemFlags tab_flags = (dataflow->dirty ? ImGuiTabItemFlags_UnsavedDocument : 0);
                if(pending_dataflow_ != nullptr){
                    tab_flags = tab_flags | (pending_dataflow_ == dataflow ? ImGuiTabItemFlags_SetSelected : 0);
                }




                bool visible = ImGui::BeginTabItem(dataflow->GetName(), &dataflow->open, tab_flags);

                // Cancel attempt to close when unsaved add to save queue so we can display a popup.
                if (!dataflow->open && dataflow->dirty)
                {
                    dataflow->open = true;
                    dataflow->DoQueueClose();
                }

                dataflow->DrawContextMenu();
                if (visible)
                {
                    current_dataflow_ = dataflow;
                    dataflow->Draw(height, 0);
                    ImGui::EndTabItem();
                }
            }

            ImGui::EndTabBar();
        }
    }

    ImGui::EndChild();

    pending_dataflow_ = nullptr;

    // Update closing queue
    //static ImVector<DataflowFile*> close_queue;
    static std::vector<DataflowFile*> close_queue;
    static std::stack<DataflowFile*> save_queue;
    if (close_queue.empty())
    {
        // Close queue is locked once we started a popup
        for (int doc_n = 0; doc_n < dataflow_files_.size(); doc_n++)
        {
            DataflowFile* dataflow = dataflow_files_[doc_n];
            if (dataflow->wantClose)
            {
                dataflow->wantClose = false;
                close_queue.push_back(dataflow);
            }
        }
    }

    // Display closing confirmation UI
    if (!close_queue.empty())
    {
        int close_queue_unsaved_documents = 0;
        for (int n = 0; n < close_queue.size(); n++)
            if (close_queue[n]->dirty)
                close_queue_unsaved_documents++;

        if (close_queue_unsaved_documents == 0)
        {
            // Close documents when all are clean
            for (int n = 0; n < close_queue.size(); n++)
                close_queue[n]->DoForceClose();
            close_queue.clear();
        } else {

            if (!ImGui::IsPopupOpen("Save?"))
                ImGui::OpenPopup("Save?");
            if (ImGui::BeginPopupModal("Save?", NULL, ImGuiWindowFlags_AlwaysAutoResize))
            {
                ImGui::Text("Save change to the following items?");
                float item_height = ImGui::GetTextLineHeightWithSpacing();
                if (ImGui::BeginChildFrame(ImGui::GetID("frame"), ImVec2(-FLT_MIN, 6.25f * item_height)))
                {
                    for (int n = 0; n < close_queue.size(); n++)
                        if (close_queue[n]->dirty)
                            ImGui::Text("%s", close_queue[n]->GetName());
                    ImGui::EndChildFrame();
                }

                ImVec2 button_size(ImGui::GetFontSize() * 7.0f, 0.0f);
                if (ImGui::Button("Yes", button_size))
                {
                    for (int n = 0; n < close_queue.size(); n++)
                    {
                        if (close_queue[n]->dirty){
                            close_queue[n]->DoQueueSave();
                        }
                        close_queue[n]->DoForceClose();
                    }
                    close_queue.clear();
                    ImGui::CloseCurrentPopup();
                }
                ImGui::SameLine();
                if (ImGui::Button("No", button_size))
                {
                    for (int n = 0; n < close_queue.size(); n++)
                        close_queue[n]->DoForceClose();
                    close_queue.clear();
                    ImGui::CloseCurrentPopup();
                }
                ImGui::SameLine();
                if (ImGui::Button("Cancel", button_size))
                {
                    close_queue.clear();
                    ImGui::CloseCurrentPopup();
                }
                ImGui::EndPopup();
            }
        }


    }

    if(save_queue.empty()) {
        // Save queue is locked once we started a popup
        for (int doc_n = 0; doc_n < dataflow_files_.size(); doc_n++)
        {
            DataflowFile* dataflow = dataflow_files_[doc_n];
            if (dataflow->wantSave)
            {
                dataflow->wantSave = false;
                save_queue.push(dataflow);
            }
        }
    }
    else {
        static bool show_save_dialog = false;

        auto& current = save_queue.top();
        if(!show_save_dialog){
            if(!current->filepath.has_filename()){
                ifd::FileDialog::Instance().Save("##DataflowSaveDialog", fmt::format("Save {0}", current->GetNameString()), "*.json {.json}");
                show_save_dialog = true;
            }

        }


        if (ifd::FileDialog::Instance().IsDone("##DataflowSaveDialog")) {

            if (ifd::FileDialog::Instance().HasResult()) {
                current->filepath = ifd::FileDialog::Instance().GetResult();
                spdlog::info("SAVE {1} {0}", current->filepath.string(), current->GetNameString());
                current->DoSave();
            }
            save_queue.pop();
            show_save_dialog = false;
            ifd::FileDialog::Instance().Close();
        }
    }






}

void traact::gui::TraactGuiApp::DrawPatternPanel() {
    static ImGuiTreeNodeFlags base_flags = ImGuiTreeNodeFlags_OpenOnArrow  | ImGuiTreeNodeFlags_SpanAvailWidth ;

    if(ImGui::TreeNode("Loaded Dataflow")){

        for(const auto& dataflow : dataflow_files_){
            if(!dataflow->open)
                continue;


            ImGuiTreeNodeFlags node_flags = base_flags;
            if (dataflow == current_dataflow_)
                node_flags |= ImGuiTreeNodeFlags_Selected;

            bool node_open = ImGui::TreeNodeEx(dataflow, node_flags, "%s", dataflow->GetName());
            if (ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left) && ImGui::IsItemHovered()) {
                if(current_dataflow_ != dataflow)
                    pending_dataflow_ = dataflow;
            }

            if (node_open)
            {


                for(const auto& tmp : dataflow->graph_editor_.Graph->getAll()){
                    ImGui::Text(tmp->getPatternName().c_str());
                }
                ImGui::TreePop();
            }
        }

        ImGui::TreePop();
    }
    ImGui::SetNextTreeNodeOpen(true);
    if(ImGui::TreeNode("All Patterns")){
        int i = 0;
        for (const auto& tmp : available_patterns_) {

            //ImGui::PushID(i);

            ImGui::Button(tmp->name.c_str());
            if(ImGui::BeginDragDropSource(ImGuiDragDropFlags_None)){

                ImGui::SetDragDropPayload("NEW_PATTERN_DRAGDROP", tmp->name.c_str(), tmp->name.length()+1);


                ImGui::EndDragDropSource();
            }


            //ImGui::PopID();
            i++;

        }


        ImGui::TreePop();
    }


}

void traact::gui::TraactGuiApp::DrawDetailsPanel() {
    ImGui::Text("Filename ");
    if(current_dataflow_){
        //ImGui::SameLine(4);
        ImGui::Text("%s",current_dataflow_->filepath.c_str());
    }

    ImGui::Text("Graph name");
    if(current_dataflow_){
        //ImGui::SameLine(4);
        //ImGui::InputText()
        static char password[64] = "password123";
        if(ImGui::InputText("password", password, IM_ARRAYSIZE(password), ImGuiInputTextFlags_EnterReturnsTrue)) {
          current_dataflow_->graph_editor_.Graph->name = password;
        };
        ImGui::Text("%s",current_dataflow_->GetName());
    }
}

void traact::gui::TraactGuiApp::SaveConfig() {
    try{
        nlohmann::json json_graph;

        for (const auto& file : recent_files_) {
            json_graph["RecentFiles"].push_back(file);
        }



        std::ofstream config_file;
        config_file.open(config_file_);
        config_file << json_graph.dump(4);
        config_file.close();
    } catch (...) {
        spdlog::error("error saving config file");
    }

}

void traact::gui::TraactGuiApp::LoadConfig() {
    try{
        nlohmann::json json_graph;
        std::ifstream config_file;
        config_file.open(config_file_);
        config_file >> json_graph;
        config_file.close();

        auto recent_result = json_graph.find("RecentFiles");

        if(recent_result != json_graph.end()){
            for (auto& recent_file : *recent_result) {
                recent_files_.push_back(recent_file.get<std::string>());
            }
        }


    } catch (...) {
        spdlog::error("error loading config file");
    }

}

void traact::gui::TraactGuiApp::NewFile(const std::string &dataflow_json) {
    nlohmann::json json_graph;
    std::stringstream ss(dataflow_json);
    ss >> json_graph;
    auto new_dataflow = new DataflowFile(json_graph);
    new_dataflow->DoQueueSave();
    dataflow_files_.push_back(new_dataflow);

}




