/** Copyright (C) 2022  Frieder Pankratz <frieder.pankratz@gmail.com> **/

#include <cstdint>

#include <functional>
#include "Window.h"
#include <glm/ext.hpp>
#include "traact_gui/debug_run/traact_opengl.h"
#include <spdlog/spdlog.h>

namespace traact::gui::scene {


Window::Window()  {
    glEnable(GL_PROGRAM_POINT_SIZE);

    auto camera_object = addObject("camera_object");
    camera_ = camera_object->getComponent<component::Camera>("camera");
}

void Window::draw() {
    init();

    render();

    //ImGui::Begin(window_name_, nullptr, ImGuiWindowFlags_NoBackground | ImGuiWindowFlags_NoDocking);


    glBindTexture(GL_TEXTURE_2D, render_texture_);
    ImVec2 avail_size = ImGui::GetContentRegionAvail();
    ImGui::Image(reinterpret_cast<void *>( static_cast<intptr_t>( render_texture_ )), avail_size, ImVec2(0,1), ImVec2(1,0));

    //ImGui::End();


    drawSceneSettings();

}



void Window::drawSceneSettings() {
    auto isRoot = [](const auto& value){
        return !value.second->getTransform()->getParent();
    };

    ImGui::Begin("Scene Settings", nullptr, ImGuiWindowFlags_NoDocking);
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
    static ImGuiTreeNodeFlags base_flags = ImGuiTreeNodeFlags_OpenOnArrow | ImGuiTreeNodeFlags_SpanAvailWidth;

    auto isParentOf = [parent = name_object->second->getTransform()](const auto& value){
        return value.second->getTransform()->getParent() == parent;
    };

    if(ImGui::TreeNodeEx(name_object->second.get(), base_flags, "%s", name_object->first.c_str())) {
        name_object->second->drawGui();

        auto child_object = std::find_if(objects_.begin(), objects_.end(), isParentOf);
        while(child_object != objects_.end() ){
            drawSceneSettings(child_object);
            child_object = std::find_if (++child_object, objects_.end(), isParentOf);
        }
        ImGui::TreePop();
    }


}

void Window::init() {
    if (!init_) {
        glGenFramebuffers(1, &frame_buffer_);

        glGenTextures(1, &render_texture_);
        glBindTexture(GL_TEXTURE_2D, render_texture_);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, render_width_, render_height_, 0, GL_RGBA, GL_UNSIGNED_BYTE, 0);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);

        glGenRenderbuffers(1, &depth_buffer_);
        glBindRenderbuffer(GL_RENDERBUFFER, depth_buffer_);
        glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT, render_width_, render_height_);

        init_ = true;


    }
}

void Window::render() {

    glBindRenderbuffer(GL_RENDERBUFFER, depth_buffer_);
    glBindFramebuffer(GL_FRAMEBUFFER, frame_buffer_);

    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, depth_buffer_);
    glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, render_texture_, 0);

    GLenum DrawBuffers[1] = {GL_COLOR_ATTACHMENT0};
    glDrawBuffers(1, DrawBuffers);

    const GLenum frameBufferStatus = glCheckFramebufferStatus(GL_FRAMEBUFFER);
    if (frameBufferStatus != GL_FRAMEBUFFER_COMPLETE) {

    }

    glViewport(0, 0, render_width_, render_height_);

    glEnable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glClearColor(0, 0, 0, 1);
    glClearDepth(100.0);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    glMatrixMode(GL_PROJECTION);
    glLoadMatrixf(glm::value_ptr(camera_->getVPMatrix()));


    for(auto& object : objects_){
        object.second->draw();
    }

    //drawCoordinateFrame(glm::mat4(1.0f), 1.0f);

    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glBindRenderbuffer(GL_RENDERBUFFER, 0);
}
void Window::update(buffer::ComponentBuffer &buffer, std::vector<RenderCommand> &additional_commands) {
    for(auto& object : objects_){
        object.second->update(buffer, additional_commands);
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
    auto object = std::make_shared<Object>(this);
    objects_.emplace(object_name, object);
    return object;
}

} // traact