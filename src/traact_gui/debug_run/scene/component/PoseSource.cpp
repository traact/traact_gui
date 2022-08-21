/** Copyright (C) 2022  Frieder Pankratz <frieder.pankratz@gmail.com> **/

#include "PoseSource.h"
#include "traact_gui/debug_run/traact_opengl.h"
namespace traact::gui::scene::component {
PoseSource::PoseSource(const std::shared_ptr<Object> &object,std::string name) : Component(object, std::move(name)) {

}
void PoseSource::setSource(application_data::PoseSourcePtr source) {
    pose_source_ = source;

}

void PoseSource::update() {
    // convert to euler
    const auto& opencv_pose = pose_source_->getBuffer();
    Eigen::Vector3f euler_angles = opencv_pose.rotation().eulerAngles(0, 1, 2);

    Eigen::Matrix3f rotation = (Eigen::AngleAxisf( euler_angles[0], Eigen::Vector3f::UnitX())
        * Eigen::AngleAxisf(-euler_angles[1], Eigen::Vector3f::UnitY()) // invert rotation direction around y-axis
        * Eigen::AngleAxisf(-euler_angles[2], Eigen::Vector3f::UnitZ())).matrix(); // invert rotation direction around y-axis
    Eigen::Quaternionf opengl_rotation = Eigen::Quaternionf(rotation);

    auto& translation = opencv_pose.translation();
    Eigen::Translation3f opengl_position(translation[0] ,
                                         -translation[1] ,
                                         -translation[2]
    );
    Eigen::Affine3f opengl_pose = opengl_position * opengl_rotation;
    transform_->setLocalPose(eigen2glm(opengl_pose.matrix()));
}
} // traact