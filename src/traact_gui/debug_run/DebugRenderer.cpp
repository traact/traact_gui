/** Copyright (C) 2022  Frieder Pankratz <frieder.pankratz@gmail.com> **/


#include "DebugRenderer.h"
#include "CalibrationComponent.h"
#include "ImageComponent.h"
#include "KeyPointListComponent.h"
#include "Position3DComponent.h"
#include "Pose6DComponent.h"
#include "PointCloudComponent.h"
#include <traact/component/generic/RawApplicationSyncSink.h>
#include <traact/component/generic/FileReaderWriter.h>

namespace traact::gui {
DebugRenderer::DebugRenderer() : scene_window_(std::make_shared<DebugSceneWindow>("DebugScene", this)) {
    render_components_["DebugScene"].emplace_back(scene_window_);
}

void DebugRenderer::init(DefaultFacade &facade, const std::string &debug_sink_id) {
    auto raw_sink = facade.getComponentAs<component::facade::RawApplicationSyncSink>(debug_sink_id);

    auto config_callback = [this](const auto& pattern_instance) {
        configureInstance(pattern_instance);
    };
    auto data_callback = [this](auto& data) {
        return processTimePoint(data);
    };
    raw_sink->setConfigCallback(config_callback);
    raw_sink->setCallback(data_callback);
    raw_sink->setInvalidCallback(data_callback);

    user_events_ = facade.findComponents<component::SyncUserEventComponent>();
    auto calib_reads = facade.findComponents<component::FileReaderWriterRead<spatial::Pose6DHeader>>();
    auto calib_writes = facade.findComponents<component::FileReaderWriterWrite<spatial::Pose6DHeader>>();

    for(auto& pattern_instance : calib_reads){
        addSceneObject(pattern_instance);
    }
    for(auto& pattern_instance : calib_writes){
        addSceneObject(pattern_instance);
    }
}

void DebugRenderer::draw() {


//    ImGui::PushStyleVar(ImGuiStyleVar_WindowMinSize, ImVec2(640, 480));
//    ImGui::Begin("Debug Window", nullptr, ImGuiWindowFlags_DockNodeHost);
//    ImGui::PopStyleVar();

    ImGui::PushStyleVar(ImGuiStyleVar_WindowMinSize, ImVec2(250, 250));
    bool additional_commands_processed{false};
    {
        std::unique_lock guard(command_lock_);
        for (auto &[window_name, all_commands] : render_commands_) {
            auto &additional = additional_commands_[window_name];



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
    for (const auto &port : pattern_instance.getConsumerPorts(kDefaultTimeDomain)) {

        SPDLOG_INFO("got port {0} {1} {2}", port->getName(), port->getDataType(), port->getPortIndex());
        auto port_name = port->getName();

        if (port_name.find(kDefaultName) != 0) {
            continue;
        }

        port_name.erase(0, kDefaultName.length());
        auto port_segmented = segmentPort(port_name);


        if (port_segmented.size() < 3) {
            SPDLOG_ERROR(
                "port names for debug renderer must follow the naming scheme {window/scene}_{object_id}_{purpose}");
        }

        auto window_type = port_segmented[0];
        if (window_type == "window") {
            addWindow(port_segmented, port);
        } else if (window_type == "scene") {
            addScene(port_segmented, port);
        } else {
            SPDLOG_ERROR("unknown window type {0}", window_type);
        }

    }

    for (auto &name_components : render_components_) {
        std::sort(name_components.second.begin(), name_components.second.end(),
                  [](const std::shared_ptr<DebugRenderComponent> &a,
                     const std::shared_ptr<DebugRenderComponent> &b) -> bool {
                      return a->getPriority() < b->getPriority();
                  });

        // default init
        auto &calibration = camera_calibration_[name_components.first];
        calibration.width = 640;
        calibration.height = 480;
        calibration.fx = 500;
        calibration.fy = 500;
        calibration.cx = calibration.width / 2;
        calibration.cy = calibration.height / 2;
        calibration.skew = 0;
    }

}

std::vector<std::string> DebugRenderer::segmentPort(const std::string &port_name) const {
    std::vector<std::string> port_segmented;
    std::stringstream port_stream(port_name);
    std::string segment;
    while (std::getline(port_stream, segment, '_')) {
        port_segmented.push_back(segment);
    }
    return port_segmented;
}

bool DebugRenderer::processTimePoint(buffer::ComponentBuffer &data) {

    {
        std::unique_lock guard(command_lock_);

        bool has_add{false};
        for (auto &[window_name, components] : render_components_) {
            auto &additional = additional_commands_[window_name];
            auto &commands = render_commands_[window_name];
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
    SPDLOG_DEBUG("updateMovement current render commands: done in {0}", end - start);

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
    auto &render_size = render_size_[window_name];
    auto &image_size = image_size_[window_name];

    if (image_size.has_value() && render_size.has_value()) {
        return render_size.value() / image_size.value();
    } else {
        return {1.0, 1.0};
    }
}
const vision::CameraCalibration &DebugRenderer::getCameraCalibration(const std::string &window_name) {
    return camera_calibration_[window_name];
}
void DebugRenderer::addWindow(const std::vector<std::string> &port_segmented,
                              pattern::instance::PortInstance::ConstPtr const &port) {
    auto object_id = port_segmented[1];
    auto window_name = fmt::format("Camera {0}", object_id);

    if (port_segmented[2] == "image") {
        addWindowImage(window_name, port_segmented, port);
    } else if (port_segmented[2] == "pose") {
        addWindowPose(window_name, port_segmented, port);
    } else if (port_segmented[2] == "calibration") {
        addWindowCalibration(window_name, port_segmented, port);
    } else {
        SPDLOG_ERROR("unsupported window purpose: {0}", port_segmented[2]);
    }

}
void DebugRenderer::addScene(const std::vector<std::string> &port_segmented,
                             pattern::instance::PortInstance::ConstPtr const &port) {

    scene_window_->addDebugObject(port_segmented, port);
}
void DebugRenderer::addWindowImage(const std::string &window_name,
                                   const std::vector<std::string> &port_segmented,
                                   pattern::instance::PortInstance::ConstPtr const &port) {

    if (port->getDataType() == vision::ImageHeader::NativeTypeName) {
        render_components_[window_name].emplace_back(std::make_shared<ImageComponent>(port->getPortIndex(),
                                                                                      port->getName(),
                                                                                      window_name, this));
    } else if (port->getDataType() == vision::GpuImageHeader::NativeTypeName) {

    } else {
        SPDLOG_ERROR("unsupported datatype for window image: {0}", port->getDataType());
    }
}
void DebugRenderer::addWindowCalibration(const std::string &window_name,
                                         const std::vector<std::string> &port_segmented,
                                         pattern::instance::PortInstance::ConstPtr const &port) {
    if (port->getDataType() == vision::CameraCalibrationHeader::NativeTypeName) {
        render_components_[window_name].emplace_back(std::make_shared<CalibrationComponent>(port->getPortIndex(),
                                                                                            port->getName(),
                                                                                            window_name, this));
    } else {
        SPDLOG_ERROR("unsupported datatype for window calibration: {0}", port->getDataType());
    }
}
void DebugRenderer::addWindowPose(const std::string &window_name,
                                  const std::vector<std::string> &ports_segmented,
                                  pattern::instance::PortInstance::ConstPtr const &port) {
    if (port->getDataType() == spatial::Pose6DHeader::NativeTypeName) {
        render_components_[window_name].emplace_back(std::make_shared<Pose6DComponent>(port->getPortIndex(),
                                                                                       port->getName(),
                                                                                       window_name, this));
    } else {
        SPDLOG_ERROR("unsupported datatype for window pose: {0}", port->getDataType());
    }

}
template<class T>
void DebugRenderer::addSceneObject(T& pattern_instance) {
    auto segmented_name = segmentPort(pattern_instance->getName());
    if(segmented_name.empty()){
        SPDLOG_ERROR("component name is empty");
        return;
    }
    if(segmented_name[0] != "scene"){
        SPDLOG_INFO("skipping non scene pattern: {0}", pattern_instance->getName());
        return;
    }
    scene_window_->addDebugObject(segmented_name, pattern_instance);
}
} // traact