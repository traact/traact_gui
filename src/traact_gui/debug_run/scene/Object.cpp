/** Copyright (C) 2022  Frieder Pankratz <frieder.pankratz@gmail.com> **/

#include "Object.h"
#include "Window.h"

namespace traact::gui::scene {
Object::Object(Window* window) : transform_(std::make_shared<Transform>()), window_(window){

}
void Object::init() {

}
void Object::update(buffer::ComponentBuffer &buffer, std::vector<RenderCommand> &additional_commands) {
    for(auto& component : components_){
        if(component.second->active){
            component.second->update(buffer, additional_commands);
        }

    }
}
void Object::draw() {
    for(auto& component : components_){
        if(component.second->active){
            component.second->draw();
        }
    }
}
void Object::drawGui() {
    for(auto& component : components_){
        ImGui::BeginGroup();
        ImGui::Text("%s",component.second->getName());
        ImGui::PushID(component.second.get());
        ImGui::Checkbox("Active", &component.second->active);
        ImGui::PopID();
        component.second->drawGui();
        ImGui::EndGroup();
    }
}
void Object::stop() {

}
const Transform::SharedPtr &Object::getTransform() {
    return transform_;
}
std::shared_ptr<component::Camera> Object::getMainCamera() const{
    return window_->getMainCamera();
}
void Object::addComponent(const std::string &name, Component::SharedPtr component) {
    components_.emplace(name, component);
}

} // traact