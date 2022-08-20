/** Copyright (C) 2022  Frieder Pankratz <frieder.pankratz@gmail.com> **/

#include "RenderCoordinateSystem.h"
#include "traact_gui/debug_run/traact_opengl.h"

namespace traact::gui::scene::component {
RenderCoordinateSystem::RenderCoordinateSystem(const std::shared_ptr<Object> &object,std::string name) : Component(object, std::move(name)) {}
void RenderCoordinateSystem::draw() {
    auto world_pose = transform_->getWorldPose();
    drawCoordinateFrame(transform_->getWorldPose(), scale_);
}
void RenderCoordinateSystem::drawGui() {
    ImGui::SliderFloat("Scale", &scale_, 0.0f, 10.0f);
}
} // traact