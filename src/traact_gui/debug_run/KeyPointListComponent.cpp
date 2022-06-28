/** Copyright (C) 2022  Frieder Pankratz <frieder.pankratz@gmail.com> **/

#include "KeyPointListComponent.h"
#include "DebugRenderer.h"

namespace traact {
gui::KeyPointListComponent::KeyPointListComponent(int port_index,
                                                  const std::string &port_name,
                                                  const std::string &window_name,
                                                  gui::DebugRenderer *renderer) : DebugRenderComponent(2000, port_index, port_name, window_name, renderer) {

}
void gui::KeyPointListComponent::update(buffer::ComponentBuffer &buffer,
                                        std::vector<RenderCommand> &additional_commands) {

    if(buffer.isInputValid(port_index_)){
        const auto& data = buffer.getInput<vision::KeyPointListHeader>(port_index_);
        render_command_ =[this, data]() {
            draw(data);
        };
    } else {
        render_command_ = [](){};
    }

}

void gui::KeyPointListComponent::draw(const vision::KeyPointList &data) {
    auto win_pos = ImGui::GetWindowPos()+ImGui::GetWindowContentRegionMin();
    ImVec2 scale = renderer_->getScale(window_name_);

    ImDrawList *draw_list = ImGui::GetWindowDrawList();
    for (const auto &point : data) {
        ImVec2 point_pos(point.pt.x,point.pt.y);
        point_pos = win_pos + point_pos * scale;
        ImVec2 size = ImVec2(point.size/2,point.size/2)  * scale;
        draw_list->AddQuad(point_pos+ImVec2(1,1)*size, point_pos+ImVec2(1,-1)*size, point_pos+ImVec2(-1,-1)*size, point_pos+ImVec2(-1,1)*size,ImColor(255, 0, 0), 2 );
        //draw_list->AddCircle(point_pos, point.size*scale.x, ImColor(255, 0, 0));
    }
}
} // traact