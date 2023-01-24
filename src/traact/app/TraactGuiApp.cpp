/** Copyright (C) 2022  Frieder Pankratz <frieder.pankratz@gmail.com> **/

#include "TraactGuiApp.h"

#include <traact/facade/DefaultFacade.h>
#include <traact/serialization/JsonGraphInstance.h>
#include <fstream>
#include <utility>

#include "traact/ImGuiUtils.h"
#include <implot.h>
#include "external/ImFileDialog/ImFileDialog.h"
#include "external/imgui-node-editor/imgui_node_editor.h"

traact::gui::TraactGuiApp::TraactGuiApp() {


//    details_editor_.onChange = [this](const std::string& instance_id) {
//        current_dataflow_->dirty = true;
//        debug_run_.parameterChanged(instance_id);
//    };


}

traact::gui::TraactGuiApp::~TraactGuiApp() {

}


void traact::gui::TraactGuiApp::render() {


    for(auto i = 0;i<windows_.size();++i){
        if(render_windows_[i]){
            windows_[i]->render();
        }
    }
    menuBar();

}

bool traact::gui::TraactGuiApp::renderStop() {
    // return true until it is ok to close all windows and stop the app
    bool keep_running = false;
    for(auto i = 0;i<windows_.size();++i){
        bool window_keep_running = windows_[i]->renderStop();
        keep_running = keep_running || window_keep_running;
    }
    return keep_running;
}

void traact::gui::TraactGuiApp::menuBar() {
    auto &gui_io = ImGui::GetIO();

    static bool show_imgui_demo = false;

    if (ImGui::BeginMainMenuBar()) {
        if (ImGui::BeginMenu("Window")) {
            for(auto i=0;i<windows_.size(); ++i){
                bool tmp = render_windows_[i];
                if(ImGui::Checkbox(windows_[i]->name().c_str(), &tmp)) {
                    render_windows_[i] = tmp;
                };
            }

            ImGui::EndMenu();
        }

        if (ImGui::BeginMenu("Help")) {
            if (ImGui::MenuItem("ImGui Demo Window")) {
                show_imgui_demo = !show_imgui_demo;
            }
            ImGui::EndMenu();
        }
        ImGui::EndMainMenuBar();
    }




    if (show_imgui_demo) {
        ImGui::ShowDemoWindow(&show_imgui_demo);
    }
}

void traact::gui::TraactGuiApp::onComponentPropertyChange() {

}
void traact::gui::TraactGuiApp::addWindow(traact::gui::Window::SharedPtr window) {
    windows_.emplace_back(std::move(window));
    render_windows_.emplace_back(true);

}
void traact::gui::TraactGuiApp::init() {
    for(auto i = 0;i<windows_.size();++i){
        windows_[i]->init();
    }

}




