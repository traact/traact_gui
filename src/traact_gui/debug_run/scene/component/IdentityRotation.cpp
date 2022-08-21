/** Copyright (C) 2022  Frieder Pankratz <frieder.pankratz@gmail.com> **/

#include "IdentityRotation.h"

namespace traact::gui::scene::component  {
IdentityRotation::IdentityRotation(const std::shared_ptr<Object> &object, const std::string &name) : Component(object,
                                                                                                               name) {}
void IdentityRotation::update() {



    glm::mat4 local_pose = transform_->getLocalPose();



    auto parent = transform_->getParent();
    if(parent) {
        glm::mat4 world_pose(1);
        world_pose[3] = transform_->getWorldPose()[3];
        auto parent_pose = parent->getWorldPose();
        glm::mat4  diff = glm::inverse(parent_pose) * world_pose;
        diff[3] = local_pose[3];
        local_pose = diff;
    }

    transform_->setLocalPose(local_pose);
}
} // traact