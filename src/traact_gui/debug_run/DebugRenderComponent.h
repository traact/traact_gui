/** Copyright (C) 2022  Frieder Pankratz <frieder.pankratz@gmail.com> **/

#ifndef TRAACT_GUI_SRC_TRAACT_GUI_DEBUG_RUN_DEBUGRENDERCOMPONENT_H_
#define TRAACT_GUI_SRC_TRAACT_GUI_DEBUG_RUN_DEBUGRENDERCOMPONENT_H_
#include <traact/traact.h>
#include <traact/vision.h>

namespace traact::gui {
using RenderCommand = std::function<void(void)>;

class DebugRenderer;

class DebugRenderComponent {
 public:
    DebugRenderComponent(int priority,
                         int port_index,
                         const std::string &port_name,
                         std::string window_name,
                         DebugRenderer *renderer);
    virtual ~DebugRenderComponent() = default;
    int getPriority();

    virtual void update(buffer::ComponentBuffer &buffer, std::vector<RenderCommand> &additional_commands) = 0;

    RenderCommand getNextCommand();
 protected:
    int priority_;
    int port_index_;
    std::string port_name_;
    std::string window_name_;
    DebugRenderer* renderer_;

    RenderCommand render_command_;
};

} // traact

#endif //TRAACT_GUI_SRC_TRAACT_GUI_DEBUG_RUN_DEBUGRENDERCOMPONENT_H_
