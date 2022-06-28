/** Copyright (C) 2022  Frieder Pankratz <frieder.pankratz@gmail.com> **/

#include <traact/math/perspective.h>
#include "Position3DComponent.h"
#include "DebugRenderer.h"

namespace traact::gui {
Position3DComponent::Position3DComponent(int port_index,
                                         const std::string &port_name,
                                         const std::string &window_name,
                                         DebugRenderer *renderer) : DebugRenderComponent(3000, port_index, port_name, window_name, renderer){

}
void Position3DComponent::update(buffer::ComponentBuffer &buffer, std::vector<RenderCommand> &additional_commands) {
    if(buffer.isInputValid(port_index_)){
        const auto& data = buffer.getInput<vision::Position3DListHeader>(port_index_);
        render_command_ =[this, data]() {
            draw(data);
        };
    } else {
        render_command_ = [](){};
    }
}

void Position3DComponent::draw(const vision::Position3DList &data) {
    auto win_pos = ImGui::GetWindowPos()+ImGui::GetWindowContentRegionMin();
    ImVec2 scale = renderer_->getScale(window_name_);
    const auto& calibration = renderer_->getCameraCalibration(window_name_);

    ImDrawList *draw_list = ImGui::GetWindowDrawList();
    for (const auto &point : data) {
        auto p_image = traact::math::reproject_point(calibration, point);
        ImVec2 point_pos(p_image.x,p_image.y);
        point_pos = win_pos + point_pos * scale;
        draw_list->AddCircle(point_pos, 7, ImColor(0, 84, 240));
    }
}
} // traact