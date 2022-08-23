/** Copyright (C) 2022  Frieder Pankratz <frieder.pankratz@gmail.com> **/

#include <traact/spatial_convert.h>
#include "StaticPose.h"

namespace traact::gui::scene::component  {
StaticPose::StaticPose(const std::shared_ptr<Object> &object, const std::string &name) : Component(object, name) {}
void StaticPose::setPattern(const std::shared_ptr<traact::pattern::instance::PatternInstance> &pattern_instance) {
    pattern_instance_ = pattern_instance;
}
void StaticPose::drawGui() {
    if(ImGui::Button("Save")){
        saveCalibration();
    }
}
void StaticPose::saveCalibration() {
    auto eigen_opengl = Eigen::Map<const Eigen::Matrix4f>(glm::value_ptr(transform_->getLocalPose()));
    auto opengl_pose = Eigen::Affine3f(eigen_opengl);
    spatial::Pose6D traact_pose = spatial::convert<spatial::OpenGLCoordinateSystem, spatial::TraactCoordinateSystem>(opengl_pose);

    spatial::Rotation3D rotation(traact_pose.rotation());

    pattern_instance_->setParameter("tx", traact_pose.translation().x());
    pattern_instance_->setParameter("ty", traact_pose.translation().y());
    pattern_instance_->setParameter("tz", traact_pose.translation().z());

    pattern_instance_->setParameter("rx", rotation.x());
    pattern_instance_->setParameter("ry", rotation.y());
    pattern_instance_->setParameter("rz", rotation.z());
    pattern_instance_->setParameter("rw", rotation.w());
}
} // traact