/** Copyright (C) 2022  Frieder Pankratz <frieder.pankratz@gmail.com> **/

#ifndef TRAACT_GUI_SRC_TRAACT_GUI_DEBUG_RUN_SCENEWINDOW_H_
#define TRAACT_GUI_SRC_TRAACT_GUI_DEBUG_RUN_SCENEWINDOW_H_

#include "imgui.h"
#define IMGUI_DEFINE_MATH_OPERATORS
#include "imgui_internal.h"

#include "ImGuizmo.h"
#include <math.h>
#include <vector>
#include <algorithm>
#include <GL/glew.h>
#include "traact_gui/opengl_shader.h"

#include <unsupported/Eigen/OpenGLSupport>
#include <glm/glm.hpp>
#include "traact_gui/debug_run/scene/component/Camera.h"

#include "traact_gui/debug_run/DebugRenderComponent.h"

#include "Component.h"
#include "Object.h"
#include <ImGuizmo.h>
namespace traact::gui::scene {

class Window {
 public:
    Window();
    void update();
    void draw();
    [[nodiscard]] std::shared_ptr<traact::gui::scene::component::Camera> getMainCamera() const;
    Object::SharedPtr findObject(const std::string& object_name) const;
    Object::SharedPtr addObject(const std::string& object_name);
 private:

    constexpr static const int render_width_{1280};
    constexpr static const int render_height_{720};

    std::shared_ptr<scene::component::Camera> camera_;


    bool init_{false};
    GLuint frame_buffer_ = 0;
    GLuint render_texture_;
    GLuint depth_buffer_;

    std::map<std::string, scene::Object::SharedPtr> objects_;

    bool render_grid_{true};
    ImGuizmo::OPERATION current_gizmo_operation_{ImGuizmo::OPERATION::TRANSLATE};
    ImGuizmo::MODE current_gizmo_mode_{ImGuizmo::LOCAL};
    int window_flags_{ImGuiWindowFlags_NoDocking};
    scene::Object::SharedPtr current_gizmo_object_;


    void render();
    void init();

    void drawSceneGraph();
    void drawSceneSettings(std::map<std::string, scene::Object::SharedPtr>::iterator  & name_object);
    void draw_edit_transform();
    void updateWindowFlags();
};



} // traact

#endif //TRAACT_GUI_SRC_TRAACT_GUI_DEBUG_RUN_SCENEWINDOW_H_
