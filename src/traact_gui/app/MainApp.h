/** Copyright (C) 2022  Frieder Pankratz <frieder.pankratz@gmail.com> **/

#ifndef TRAACT_GUI_SRC_TRAACT_GUI_MAINAPP_H_
#define TRAACT_GUI_SRC_TRAACT_GUI_MAINAPP_H_

#include <imgui.h>
#define IMGUI_DEFINE_MATH_OPERATORS
#include <imgui_internal.h>
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#include "traact_gui/opengl_shader.h"
#include "traact_gui/file_manager.h"
#include <stdio.h>
#include <iostream>
#include <vector>
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include "external/ImFileDialog/ImFileDialog.h"
# include "external/imgui-node-editor/imgui_node_editor.h"
#include <spdlog/spdlog.h>

#include "TraactGuiApp.h"


namespace traact::gui {
class MainApp {
 public:
    MainApp(std::atomic_bool &should_stop);
    ~MainApp();

    void blockingLoop();
    void loadDataflow(std::string dataflow_file);
 private:
    GLFWwindow *window_;
    TraactGuiApp traact_app_{"traact_gui_config.json"};
    std::string glsl_version_;
    int screen_width_{0}, screen_height_{0};
    std::atomic_bool& should_stop_;
    bool running_{true};

    void init();
    void initFileDialog() const;
    void initOpenGl();
    void initImGui() const;
    void dispose() const;
};
}



#endif //TRAACT_GUI_SRC_TRAACT_GUI_MAINAPP_H_
