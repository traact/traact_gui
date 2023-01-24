/** Copyright (C) 2022  Frieder Pankratz <frieder.pankratz@gmail.com> **/

#ifndef TRAACT_GUI_SRC_TRAACT_DEBUG_RUN_POSE6DCOMPONENT_H_
#define TRAACT_GUI_SRC_TRAACT_DEBUG_RUN_POSE6DCOMPONENT_H_

#include "DebugRenderComponent.h"
#include <traact/vision.h>

namespace traact::gui {
class Pose6DComponent : public DebugRenderComponent{
 public:
    Pose6DComponent(int port_index,
                        const std::string &port_name,
                        const std::string &window_name,
                        DebugRenderer *renderer);
    virtual ~Pose6DComponent() = default;

    void update(buffer::ComponentBuffer &buffer, std::vector<RenderCommand> &additional_commands) override;
 private:

    void draw(const spatial::Pose6D & data);
};
}



#endif //TRAACT_GUI_SRC_TRAACT_DEBUG_RUN_POSE6DCOMPONENT_H_
