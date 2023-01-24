/** Copyright (C) 2022  Frieder Pankratz <frieder.pankratz@gmail.com> **/

#include <cstdint>

#include <functional>
#include "Window.h"
#include <glm/ext.hpp>
#include "traact/opengl/traact_opengl.h"
#include <spdlog/spdlog.h>

namespace traact::gui::scene {


Window::Window()  {
    glEnable(GL_PROGRAM_POINT_SIZE);

    auto camera_object = addObject("camera_object");
    camera_ = camera_object->getComponent<component::Camera>("camera");
}

void Window::draw() {
    ImGui::Begin("Scene", nullptr, window_flags_);
    init();

    render();

    ImGuizmo::BeginFrame();

    glBindTexture(GL_TEXTURE_2D, framebuffer_.getRenderTexture());
    ImVec2 avail_size = ImGui::GetContentRegionAvail();
    ImVec2 p = ImGui::GetCursorScreenPos();
    ImGui::Image(reinterpret_cast<void *>( static_cast<intptr_t>( framebuffer_.getRenderTexture() )), avail_size, ImVec2(0,1), ImVec2(1,0));

    static const float identityMatrix[16] =
        { 1.f, 0.f, 0.f, 0.f,
          0.f, 1.f, 0.f, 0.f,
          0.f, 0.f, 1.f, 0.f,
          0.f, 0.f, 0.f, 1.f };
    ImGui::SetCursorScreenPos(p);
    ImGuizmo::SetOrthographic(false);
    ImGuizmo::SetDrawlist();

    static ImGuiWindowFlags gizmo_window_flags = 0;
    float window_width = (float)ImGui::GetWindowWidth();
    float window_height = (float)ImGui::GetWindowHeight();
    ImGuizmo::SetRect(ImGui::GetWindowPos().x, ImGui::GetWindowPos().y, window_width, window_height);
    updateWindowFlags();

    if(render_grid_){
        ImGuizmo::DrawGrid(camera_->getViewMatrixPtr(), camera_->getProjectionMatrixPtr(),identityMatrix , 100.f);
    }

    draw_edit_transform();
    drawSceneGraph();

    ImGui::End();
}



void Window::drawSceneGraph() {
    auto isRoot = [](const auto& value){
        return !value.second->getTransform()->getParent();
    };

    ImGui::Begin("Scene Graph", nullptr, ImGuiWindowFlags_NoDocking);



    auto root_object = std::find_if(objects_.begin(), objects_.end(), isRoot);
    ImGui::SetNextItemOpen(true);
    if(ImGui::TreeNode("Root"))  {
        while(root_object != objects_.end() ){
            drawSceneSettings(root_object);
            root_object = std::find_if (++root_object, objects_.end(), isRoot);
        }
        ImGui::TreePop();
    }
    ImGui::End();

}

void Window::drawSceneSettings(std::map<std::string, scene::Object::SharedPtr>::iterator & name_object) {
    static ImGuiTreeNodeFlags base_flags = ImGuiTreeNodeFlags_OpenOnArrow | ImGuiTreeNodeFlags_OpenOnDoubleClick | ImGuiTreeNodeFlags_SpanAvailWidth;



    auto isParentOf = [parent = name_object->second->getTransform()](const auto& value){
        return value.second->getTransform()->getParent() == parent;
    };

    int gizmo_count{0};

    ImGuiTreeNodeFlags node_flags = base_flags;
    const bool is_selected = current_gizmo_object_ == name_object->second;
    if (is_selected){
        node_flags |= ImGuiTreeNodeFlags_Selected;
    }


    if(ImGui::TreeNodeEx(name_object->second.get(), node_flags, "%s", name_object->first.c_str())) {
        if (ImGui::IsItemClicked()){
            current_gizmo_object_ = name_object->second;
        }

        auto child_object = std::find_if(objects_.begin(), objects_.end(), isParentOf);
        while(child_object != objects_.end() ){
            drawSceneSettings(child_object);
            child_object = std::find_if (++child_object, objects_.end(), isParentOf);
        }
        ImGui::TreePop();
    } else {
        if (ImGui::IsItemClicked()){
            current_gizmo_object_ = name_object->second;
        }
    }


}
void Window::draw_edit_transform() {

    ImGui::Begin("Scene Edit", nullptr);
    ImGui::Checkbox("Render Grid", &render_grid_);
    ImGui::Separator();
    if(current_gizmo_object_){
        ImGui::Text("Transform %s",current_gizmo_object_->getName().c_str());
        float* matrix = current_gizmo_object_->getTransform()->getLocalPosePtr();
        if (ImGui::RadioButton("Translate", current_gizmo_operation_ == ImGuizmo::TRANSLATE)){
            current_gizmo_operation_ = ImGuizmo::OPERATION::TRANSLATE;
        }
        ImGui::SameLine();
        if (ImGui::RadioButton("Rotate", current_gizmo_operation_ == ImGuizmo::ROTATE)){
            current_gizmo_operation_ = ImGuizmo::ROTATE;
        }
        ImGui::SameLine();
        if (ImGui::RadioButton("Scale", current_gizmo_operation_ == ImGuizmo::SCALE)) {
            current_gizmo_operation_ = ImGuizmo::SCALE;
        }

        float matrixTranslation[3], matrixRotation[3], matrixScale[3];
        ImGuizmo::DecomposeMatrixToComponents(matrix, matrixTranslation, matrixRotation, matrixScale);
        ImGui::SliderFloat3("#Tr", matrixTranslation, -10.0, 10.0);
        ImGui::SliderFloat3("#Rt", matrixRotation, -180.0, 180.0);
        ImGui::SliderFloat3("#Sc", matrixScale, 0.0, 2.0);
        ImGuizmo::RecomposeMatrixFromComponents(matrixTranslation, matrixRotation, matrixScale, matrix);
        if (current_gizmo_operation_ != ImGuizmo::SCALE)
        {
            if (ImGui::RadioButton("#Local", current_gizmo_mode_ == ImGuizmo::LOCAL))
                current_gizmo_mode_ = ImGuizmo::LOCAL;
            ImGui::SameLine();
            if (ImGui::RadioButton("#World", current_gizmo_mode_ == ImGuizmo::WORLD))
                current_gizmo_mode_ = ImGuizmo::WORLD;
        }

        ImGui::Separator();
        current_gizmo_object_->drawGui();



        auto view_matrix = camera_->getViewMatrix();

        auto parent = current_gizmo_object_->getTransform()->getParent();
        if(parent){
            view_matrix = view_matrix * parent->getWorldPose();
        }


        ImGuizmo::Manipulate(
            glm::value_ptr(view_matrix), camera_->getProjectionMatrixPtr(),
            current_gizmo_operation_,
            current_gizmo_mode_, matrix, NULL, NULL, NULL, NULL);

    } else {
        ImGui::Text("Transform");
        ImGui::Text("No Object selected");
    }



    ImGui::End();

}

void Window::init() {
    if (!init_) {

        ImGuizmo::SetImGuiContext(ImGui::GetCurrentContext());
        framebuffer_.setViewPort(render_width_,render_height_);
        framebuffer_.init();

//        glGenFramebuffers(1, &frame_buffer_);
//
//        glGenTextures(1, &render_texture_);
//        glBindTexture(GL_TEXTURE_2D, render_texture_);
//        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, render_width_, render_height_, 0, GL_RGBA, GL_UNSIGNED_BYTE, 0);
//        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
//        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
//
//        glGenRenderbuffers(1, &depth_buffer_);
//        glBindRenderbuffer(GL_RENDERBUFFER, depth_buffer_);
//        glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT, render_width_, render_height_);

        init_ = true;


    }
}

void Window::render() {

    framebuffer_.bind();

//    glBindRenderbuffer(GL_RENDERBUFFER, depth_buffer_);
//    glBindFramebuffer(GL_FRAMEBUFFER, frame_buffer_);
//
//    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, depth_buffer_);
//    glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, render_texture_, 0);
//
//    GLenum DrawBuffers[1] = {GL_COLOR_ATTACHMENT0};
//    glDrawBuffers(1, DrawBuffers);
//
//    const GLenum frameBufferStatus = glCheckFramebufferStatus(GL_FRAMEBUFFER);
//    if (frameBufferStatus != GL_FRAMEBUFFER_COMPLETE) {
//
//    }

    glViewport(0, 0, render_width_, render_height_);

    glEnable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glClearColor(0, 0, 0, 1);
    glClearDepth(100.0);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    glMatrixMode(GL_PROJECTION);
    glLoadMatrixf(glm::value_ptr(camera_->getProjectionMatrix()));


    for(auto& object : objects_){
        object.second->draw();
    }

    //drawCoordinateFrame(glm::mat4(1.0f), 1.0f);

    framebuffer_.resetBind();
}
void Window::update() {
    for(auto& object : objects_){
        object.second->update();
    }
}
std::shared_ptr<component::Camera> Window::getMainCamera() const{
    return camera_;
}
Object::SharedPtr Window::findObject(const std::string &object_name) const {
    auto object = objects_.find(object_name);
    if(object == objects_.end()){
        return nullptr;
    } else {
        return object->second;
    }
}
Object::SharedPtr Window::addObject(const std::string &object_name) {
    auto object = std::make_shared<Object>(object_name, this);
    objects_.emplace(object_name, object);
    return object;
}
Object::SharedPtr Window::getObject(const std::string &object_name) {
    auto object = findObject(object_name);
    if(!object){
        object = addObject(object_name);
    }
    return object;
}
void Window::updateWindowFlags() {
    if(ImGuizmo::IsOver() || ImGuizmo::IsUsing()) {
        window_flags_ =  ImGuiWindowFlags_NoMove;
    } else {
        window_flags_ = ImGuiWindowFlags_NoDocking;
    };

}

} // traact