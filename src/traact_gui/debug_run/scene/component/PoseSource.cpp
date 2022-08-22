/** Copyright (C) 2022  Frieder Pankratz <frieder.pankratz@gmail.com> **/

#include "PoseSource.h"
#include "traact_gui/debug_run/traact_opengl.h"
#include <traact/spatial_convert.h>
namespace traact::gui::scene::component {
PoseSource::PoseSource(const std::shared_ptr<Object> &object,std::string name) : Component(object, std::move(name)) {

}
void PoseSource::setSource(application_data::PoseSourcePtr source) {
    pose_source_ = source;

}

void PoseSource::update() {

    const auto& traact_pose = pose_source_->getBuffer();

    Eigen::Affine3f opengl_pose = spatial::convert<spatial::TraactCoordinateSystem, spatial::OpenGLCoordinateSystem>(traact_pose);

    transform_->setLocalPose(eigen2glm(opengl_pose.matrix()));
}
} // traact