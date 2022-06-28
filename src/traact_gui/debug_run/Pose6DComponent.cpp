/** Copyright (C) 2022  Frieder Pankratz <frieder.pankratz@gmail.com> **/

#include <traact/math/perspective.h>
#include "Pose6DComponent.h"
#include "DebugRenderer.h"

namespace traact::gui {

Pose6DComponent::Pose6DComponent(int port_index,
                                 const std::string &port_name,
                                 const std::string &window_name,
                                 DebugRenderer *renderer) : DebugRenderComponent(4000, port_index, port_name, window_name, renderer){

}
void Pose6DComponent::update(buffer::ComponentBuffer &buffer, std::vector<RenderCommand> &additional_commands) {
    if(buffer.isInputValid(port_index_)){
        const auto& data = buffer.getInput<spatial::Pose6DHeader>(port_index_);
        render_command_ =[this, data]() {
            draw(data);
        };
    } else {
        render_command_ = [](){};
    }
}

void Pose6DComponent::draw(const spatial::Pose6D &data) {
    auto win_pos = ImGui::GetWindowPos()+ImGui::GetWindowContentRegionMin();
    ImVec2 scale = renderer_->getScale(window_name_);
    const auto& calibration = renderer_->getCameraCalibration(window_name_);

    ImDrawList *draw_list = ImGui::GetWindowDrawList();


    auto p0 = traact::math::reproject_point(data, calibration,
                                            vision::Position3D (0, 0, 0));
    auto px = traact::math::reproject_point(data, calibration,
                                            vision::Position3D (1, 0, 0));
    auto py = traact::math::reproject_point(data, calibration,
                                            vision::Position3D (0, 1, 0));
    auto pz = traact::math::reproject_point(data, calibration,
                                            vision::Position3D (0, 0, 1));
    ImVec2 p0imgui(p0.x,p0.y);
    ImVec2 pximgui(px.x,px.y);
    ImVec2 pyimgui(py.x,py.y);
    ImVec2 pzimgui(pz.x,pz.y);

    ImVec2 p_0 = win_pos + p0imgui * scale;

    draw_list->AddLine(p_0,
                       win_pos + pximgui * scale,
                       ImColor(255, 0, 0),
                       2);
    draw_list->AddLine(p_0,
                       win_pos + pyimgui * scale,
                       ImColor(0, 255, 0),
                       2);
    draw_list->AddLine(p_0,
                       win_pos + pzimgui * scale,
                       ImColor(0, 0, 255),
                       2);
}
}