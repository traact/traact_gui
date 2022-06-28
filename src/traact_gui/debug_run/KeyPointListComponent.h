/** Copyright (C) 2022  Frieder Pankratz <frieder.pankratz@gmail.com> **/

#ifndef TRAACT_GUI_SRC_TRAACT_GUI_DEBUG_RUN_KEYPOINTLISTCOMPONENT_H_
#define TRAACT_GUI_SRC_TRAACT_GUI_DEBUG_RUN_KEYPOINTLISTCOMPONENT_H_

#include "DebugRenderComponent.h"
#include <traact/vision.h>

namespace traact::gui {

class KeyPointListComponent : public DebugRenderComponent{
 public:
    KeyPointListComponent(int port_index,
                         const std::string &port_name,
                         const std::string &window_name,
                         DebugRenderer *renderer);
    virtual ~KeyPointListComponent() = default;

    void update(buffer::ComponentBuffer &buffer, std::vector<RenderCommand> &additional_commands) override;

 private:
    void draw(const vision::KeyPointList & data);

};

} // traact

#endif //TRAACT_GUI_SRC_TRAACT_GUI_DEBUG_RUN_KEYPOINTLISTCOMPONENT_H_
