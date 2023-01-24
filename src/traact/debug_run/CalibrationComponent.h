/** Copyright (C) 2022  Frieder Pankratz <frieder.pankratz@gmail.com> **/

#ifndef TRAACT_GUI_SRC_TRAACT_DEBUG_RUN_CALIBRATIONCOMPONENT_H_
#define TRAACT_GUI_SRC_TRAACT_DEBUG_RUN_CALIBRATIONCOMPONENT_H_

#include "DebugRenderComponent.h"
#include <traact/vision.h>
namespace traact::gui {

class CalibrationComponent : public DebugRenderComponent{
 public:
    CalibrationComponent(int port_index,
                   const std::string &port_name,
                   const std::string &window_name,
                   DebugRenderer *renderer);
    virtual ~CalibrationComponent() = default;

    void update(buffer::ComponentBuffer &buffer, std::vector<RenderCommand> &additional_commands) override;

};

} // traact

#endif //TRAACT_GUI_SRC_TRAACT_DEBUG_RUN_CALIBRATIONCOMPONENT_H_
