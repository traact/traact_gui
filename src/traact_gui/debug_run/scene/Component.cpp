/** Copyright (C) 2022  Frieder Pankratz <frieder.pankratz@gmail.com> **/

#include "Component.h"
#include "Object.h"
namespace traact::gui::scene {
Component::Component(std::shared_ptr<Object> object, std::string name)
    : object_(std::move(object)), transform_(object_->getTransform()), camera_(object_->getMainCamera()), name_(name) {}
void Component::update(buffer::ComponentBuffer &buffer, std::vector<RenderCommand> &additional_commands) {

}
void Component::draw() {

}
void Component::drawGui() {

}
const char *Component::getName() const {
    return name_.c_str();
}
} // traact