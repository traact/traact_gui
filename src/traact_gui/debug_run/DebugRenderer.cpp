/** Copyright (C) 2022  Frieder Pankratz <frieder.pankratz@gmail.com> **/

#include "DebugRenderer.h"
#include "ImageComponent.h"


namespace traact::gui {
void DebugRenderer::draw() {

    bool additional_commands_processed{false};
    {
        std::unique_lock guard(command_lock_);
        for (auto& [window_name, all_commands] : render_commands_) {
            auto& additional = additional_commands_[window_name];

            ImGui::Begin(window_name.c_str());

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
}
void DebugRenderer::configureInstance(const pattern::instance::PatternInstance &pattern_instance) {
    const static std::string kDefaultName{"Default_0_"};
    for (const auto& port : pattern_instance.getConsumerPorts(0)) {

        SPDLOG_INFO("got port {0} {1} {2}", port->getName(), port->getDataType(), port->getPortIndex());
        auto port_name = port->getName();

        if(port_name.find(kDefaultName) != 0){
            continue;
        }

        port_name.erase(0, kDefaultName.length());
        auto pos = port_name.find("_");
        auto window_index = port_name.substr(0, pos);
        auto window_name = fmt::format("Camera {0}", window_index);
        if(port->getDataType() == vision::ImageHeader::NativeTypeName){
            render_components_[window_name].emplace_back(new ImageComponent(port->getPortIndex(),
                                                                            port->getName(),
                                                                            window_name, this));
        }
    }

    for (auto& name_components : render_components_) {
        std::sort(name_components.second.begin(), name_components.second.end(),
                  [](const std::unique_ptr<DebugRenderComponent> &a, const std::unique_ptr<DebugRenderComponent> &b) -> bool {
                      return a->getPriority() < b->getPriority();
                  });
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
} // traact