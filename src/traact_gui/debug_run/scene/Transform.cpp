/** Copyright (C) 2022  Frieder Pankratz <frieder.pankratz@gmail.com> **/

#include "Transform.h"

namespace traact::gui::scene {
glm::mat4 Transform::getLocalPose() const {
    return pose_;
}
glm::mat4 Transform::getWorldPose() const {
    if(parent_){
        return parent_->getWorldPose() * pose_;
    } else {
        return pose_;
    }
}
void Transform::setParent(Transform::SharedPtr parent) {
    parent_ = parent;
}
Transform::SharedPtr Transform::getParent() {
    return parent_;
}
Transform::~Transform() {
    parent_.reset();
}
void Transform::setLocalPose(glm::mat4 pose) {
    pose_ = pose;

}
} // traact