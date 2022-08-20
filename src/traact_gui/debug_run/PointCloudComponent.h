/** Copyright (C) 2022  Frieder Pankratz <frieder.pankratz@gmail.com> **/

#ifndef TRAACT_GUI_SRC_TRAACT_GUI_DEBUG_RUN_POINTCLOUDCOMPONENT_H_
#define TRAACT_GUI_SRC_TRAACT_GUI_DEBUG_RUN_POINTCLOUDCOMPONENT_H_

#include "DebugRenderComponent.h"
#include <traact/pointCloud.h>

namespace traact::gui {
class PointCloudComponent : public DebugRenderComponent {
 public:
    PointCloudComponent(int port_index,
                        const std::string &port_name,
                        const std::string &window_name,
                        DebugRenderer *renderer);
    virtual void update(buffer::ComponentBuffer &buffer, std::vector<RenderCommand> &additional_commands) override;

 private:

    void draw();


};
}

#endif //TRAACT_GUI_SRC_TRAACT_GUI_DEBUG_RUN_POINTCLOUDCOMPONENT_H_
