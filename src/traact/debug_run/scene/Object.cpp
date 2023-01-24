/** Copyright (C) 2022  Frieder Pankratz <frieder.pankratz@gmail.com> **/

#include "Object.h"
#include "Window.h"

namespace traact::gui::scene {
Object::Object(const std::string &name, Window *window) : transform_(std::make_shared<Transform>()), window_(window), name_(name){

}
void Object::init() {

}
void Object::update() {
    for(auto& component : components_){
        if(component.second->active){
            component.second->update();
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

        //ImGui::BeginChild(ImGui::GetID(component.second.get()), ImVec2(0,260), true);

        ImGui::Text("%s",component.second->getName());
        ImGui::SameLine();
        ImGui::PushID(&component.second->active);
        ImGui::Checkbox("Active", &component.second->active);
        ImGui::PopID();
        component.second->drawGui();
        ImGui::Separator();
        //ImGui::EndChild();
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
const std::string &Object::getName() const {
    return name_;
}

} // traact