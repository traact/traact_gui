/** Copyright (C) 2022  Frieder Pankratz <frieder.pankratz@gmail.com> **/

#include "DebugRenderComponent.h"

namespace traact::gui {
int DebugRenderComponent::getPriority() {
    return priority_;
}
DebugRenderComponent::DebugRenderComponent(int priority,
                                           int port_index,
                                           const std::string &port_name,
                                           std::string window_name,
                                           DebugRenderer *renderer) : priority_(
    priority), port_index_(port_index), port_name_(port_name), renderer_(renderer), window_name_(window_name) {}
} // traact