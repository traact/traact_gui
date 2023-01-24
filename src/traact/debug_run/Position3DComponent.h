/** Copyright (C) 2022  Frieder Pankratz <frieder.pankratz@gmail.com> **/

#ifndef TRAACT_GUI_SRC_TRAACT_DEBUG_RUN_POSITION3DCOMPONENT_H_
#define TRAACT_GUI_SRC_TRAACT_DEBUG_RUN_POSITION3DCOMPONENT_H_

#include "DebugRenderComponent.h"
#include <traact/vision.h>

namespace traact::gui {

class Position3DComponent : public DebugRenderComponent{
 public:
    Position3DComponent(int port_index,
                          const std::string &port_name,
                          const std::string &window_name,
                          DebugRenderer *renderer);
    virtual ~Position3DComponent() = default;

    void update(buffer::ComponentBuffer &buffer, std::vector<RenderCommand> &additional_commands) override;
 private:
    void draw(const vision::Position3DList & data);
};

} // traact

#endif //TRAACT_GUI_SRC_TRAACT_DEBUG_RUN_POSITION3DCOMPONENT_H_
