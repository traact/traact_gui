/** Copyright (C) 2022  Frieder Pankratz <frieder.pankratz@gmail.com> **/

#include "DebugRenderer.h"
#include "CalibrationComponent.h"
#include "ImageComponent.h"
#include "KeyPointListComponent.h"
#include "Position3DComponent.h"
#include "Pose6DComponent.h"


namespace traact::gui {
void DebugRenderer::draw() {


//    ImGui::PushStyleVar(ImGuiStyleVar_WindowMinSize, ImVec2(640, 480));
//    ImGui::Begin("Debug Window", nullptr, ImGuiWindowFlags_DockNodeHost);
//    ImGui::PopStyleVar();

    ImGui::PushStyleVar(ImGuiStyleVar_WindowMinSize, ImVec2(250, 250));
    bool additional_commands_processed{false};
    {
        std::unique_lock guard(command_lock_);
        for (auto& [window_name, all_commands] : render_commands_) {
            auto& additional = additional_commands_[window_name];

            ImGui::Begin(window_name.c_str(), nullptr, ImGuiWindowFlags_NoDocking);

            if (!additional.empty()) {
                for (auto &command : additional) {
                    command();
                }
                additional.clear();
                additional_commands_processed = true;
            }

            for (auto &command : all_commands) {
                command();
            }
            ImGui::End();
        }

        if (additional_commands_processed) {
            SPDLOG_DEBUG("render command ts: additional_commands_processed_");
            additional_commands_processed_.SetInit(true);
        }
    }
    ImGui::PopStyleVar();

   // ImGui::End();
}
void DebugRenderer::configureInstance(const pattern::instance::PatternInstance &pattern_instance) {
    const static std::string kDefaultName{"Default_0_"};
    for (const auto& port : pattern_instance.getConsumerPorts(kDefaultTimeDomain)) {

        SPDLOG_INFO("got port {0} {1} {2}", port->getName(), port->getDataType(), port->getPortIndex());
        auto port_name = port->getName();

        if(port_name.find(kDefaultName) != 0){
            continue;
        }

        port_name.erase(0, kDefaultName.length());
        auto pos = port_name.find("_");
        auto window_index = port_name.substr(0, pos);
        auto window_name = fmt::format("Camera {0}", window_index);
        if(port->getDataType() == vision::CameraCalibrationHeader::NativeTypeName){
            render_components_[window_name].emplace_back(new CalibrationComponent(port->getPortIndex(),
                                                                            port->getName(),
                                                                            window_name, this));
        } else if(port->getDataType() == vision::ImageHeader::NativeTypeName){
            render_components_[window_name].emplace_back(new ImageComponent(port->getPortIndex(),
                                                                            port->getName(),
                                                                            window_name, this));
        } else if(port->getDataType() == vision::KeyPointListHeader::NativeTypeName){
            render_components_[window_name].emplace_back(new KeyPointListComponent(port->getPortIndex(),
                                                                            port->getName(),
                                                                            window_name, this));
        }else if(port->getDataType() == vision::Position3DListHeader::NativeTypeName){
            render_components_[window_name].emplace_back(new Position3DComponent(port->getPortIndex(),
                                                                            port->getName(),
                                                                            window_name, this));
        }else if(port->getDataType() == spatial::Pose6DHeader::NativeTypeName){
            render_components_[window_name].emplace_back(new Pose6DComponent(port->getPortIndex(),
                                                                            port->getName(),
                                                                            window_name, this));
        }
    }

    for (auto& name_components : render_components_) {
        std::sort(name_components.second.begin(), name_components.second.end(),
                  [](const std::unique_ptr<DebugRenderComponent> &a, const std::unique_ptr<DebugRenderComponent> &b) -> bool {
                      return a->getPriority() < b->getPriority();
                  });

        auto& calibration = camera_calibration_[name_components.first];
        calibration.width = 640;
        calibration.height = 480;
        calibration.fx = 500;
        calibration.fy = 500;
        calibration.cx = calibration.width/2;
        calibration.cy = calibration.height/2;
        calibration.skew = 0;
    }






}
bool DebugRenderer::processTimePoint(buffer::ComponentBuffer &data) {

    {
        std::unique_lock guard(command_lock_);

        bool has_add{false};
        for (auto& [window_name, components] : render_components_) {
            auto& additional = additional_commands_[window_name];
            auto& commands = render_commands_[window_name];
            commands.resize(components.size());
            additional.clear();
            for (int component_index = 0; component_index < components.size(); ++component_index) {
                components[component_index]->update(data, additional);
                commands[component_index] = components[component_index]->getNextCommand();
            }
            has_add = has_add || !additional.empty();
        }

        additional_commands_processed_.SetInit(!has_add);
    }


    TimestampSteady start = nowSteady();
    while (!additional_commands_processed_.tryWait()) {
        SPDLOG_WARN(
            "timeout waiting for additional render commands of render module to be processed (e.g. image texture upload)");
    }
    auto end = nowSteady();
    SPDLOG_DEBUG("update current render commands: done in {0}", end - start);

    return true;
}
void DebugRenderer::setImageSize(ImVec2 image_size, const std::string &window_name) {
    image_size_[window_name] = image_size;
}
void DebugRenderer::setImageRenderSize(ImVec2 image_size, const std::string &window_name) {
    render_size_[window_name] = image_size;

}
void DebugRenderer::setCameraCalibration(const vision::CameraCalibration &calibration, const std::string &window_name) {
    camera_calibration_[window_name] = calibration;
}
ImVec2 DebugRenderer::getScale(const std::string &window_name) {
    auto& render_size = render_size_[window_name];
    auto& image_size = image_size_[window_name];

    if(image_size.has_value() && render_size.has_value()){
        return render_size.value() / image_size.value();
    } else {
        return ImVec2(1.0,1.0);
    }
}
const vision::CameraCalibration &DebugRenderer::getCameraCalibration(const std::string &window_name) {
    return camera_calibration_[window_name];
}
} // traact