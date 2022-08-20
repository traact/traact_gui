/** Copyright (C) 2022  Frieder Pankratz <frieder.pankratz@gmail.com> **/

#include "PointCloudComponent.h"

namespace traact::gui {

void PointCloudComponent::update(buffer::ComponentBuffer &buffer, std::vector<RenderCommand> &additional_commands) {

}
PointCloudComponent::PointCloudComponent(int port_index,
                                         const std::string &port_name,
                                         const std::string &window_name,
                                         DebugRenderer *renderer) : DebugRenderComponent(1000,
                                                                                         port_index,
                                                                                         port_name,
                                                                                         window_name,
                                                                                         renderer) {}

}