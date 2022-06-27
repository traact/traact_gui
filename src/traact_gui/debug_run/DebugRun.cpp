/** Copyright (C) 2022  Frieder Pankratz <frieder.pankratz@gmail.com> **/

#include "DebugRun.h"

#include <imgui.h>
#define IMGUI_DEFINE_MATH_OPERATORS
#include <imgui_internal.h>
#include "external/imgui_misc/imgui_stdlib.h"
#include <traact/component/generic/RawApplicationSyncSink.h>

namespace traact::gui {
void DebugRun::draw() {
    ImGui::Begin("Run dataflow");
    ImGui::BeginDisabled(!canStart());
    if(ImGui::Button("Start")){
        startDataflow();
    }
    ImGui::EndDisabled();
    ImGui::SameLine();
    ImGui::BeginDisabled(!canStop());
    if(ImGui::Button("Stop")){
        facade_->stopAsync();
    }
    ImGui::EndDisabled();

    ImGui::Checkbox("Attach Debug Renderer", &attach_debug_renderer_);
    ImGui::InputText("Debug Sink Pattern ID", &debug_sink_id_);

    ImGui::End();

    if(facade_finished_.has_value() && attach_debug_renderer_){
        debug_renderer_.draw();
    }

    if(facade_finished_.has_value()){
        auto has_finished = facade_finished_.value().wait_for(std::chrono::nanoseconds(0));
        if(has_finished == std::future_status::ready){
            facade_finished_ = {};
        }
    }
}
void DebugRun::startDataflow() {
    facade_ = std::make_shared<DefaultFacade>();



    facade_->loadDataflow(current_dataflow_->graph_editor_.Graph);
    facade_finished_ = facade_->getFinishedFuture();

    connectDebugRenderer();



    std::thread start_thread([local_facade = facade_]() {
        local_facade->start();
    });
    start_thread.detach();
}
void DebugRun::connectDebugRenderer() {
    auto raw_sink = facade_->getComponentAs<component::facade::RawApplicationSyncSink>(debug_sink_id_);
    auto config_callback = [local_renderer = &debug_renderer_](const auto& pattern_instance) {
        local_renderer->configureInstance(pattern_instance);
    };
    auto data_callback = [local_renderer = &debug_renderer_](auto& data) {
        return local_renderer->processTimePoint(data);
    };
    raw_sink->setConfigCallback(config_callback);
    raw_sink->setCallback(data_callback);
    raw_sink->setInvalidCallback(data_callback);
}
void DebugRun::setCurrentDataflow(std::shared_ptr<traact::gui::DataflowFile> dataflow) {
    current_dataflow_ = dataflow;
}
bool DebugRun::canStart() {
    return current_dataflow_ != nullptr && !facade_finished_.has_value();
}
bool DebugRun::canStop() {
    return facade_ != nullptr && facade_finished_.has_value();
}
} // traact