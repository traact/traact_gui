/** Copyright (C) 2022  Frieder Pankratz <frieder.pankratz@gmail.com> **/

#include "CalibrationComponent.h"
#include "DebugRenderer.h"
#include "DebugRenderComponent.h"
namespace traact::gui {
CalibrationComponent::CalibrationComponent(int port_index,
                                           const std::string &port_name,
                                           const std::string &window_name,
                                           DebugRenderer *renderer) : DebugRenderComponent(
    500,
    port_index,
    port_name, window_name, renderer)  {

    render_command_ = [](){
    };
}
void CalibrationComponent::update(buffer::ComponentBuffer &buffer, std::vector<RenderCommand> &additional_commands) {
    if(buffer.isInputValid(port_index_)){
        const auto& calibration = buffer.getInput<vision::CameraCalibrationHeader>(port_index_);
        renderer_->setCameraCalibration(calibration, window_name_);
    }
}
} // traact