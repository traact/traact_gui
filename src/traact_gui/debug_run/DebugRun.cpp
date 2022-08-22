/** Copyright (C) 2022  Frieder Pankratz <frieder.pankratz@gmail.com> **/

#include "DebugRun.h"

#include <imgui.h>
#define IMGUI_DEFINE_MATH_OPERATORS
#include <imgui_internal.h>
#include "external/imgui_misc/imgui_stdlib.h"
#include <traact/component/generic/RawApplicationSyncSink.h>
#include <traact/util/Logging.h>


namespace traact::gui {
void DebugRun::draw() {

    ImGui::Begin("Run");
    if(current_dataflow_){
        ImGui::Text("%s",current_dataflow_->getName());
    } else {
        ImGui::Text("No dataflow selected");
    }

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

    const char* log_levels[] = {"trace", "debug", "info", "warn", "error"};

    if(ImGui::Combo("log level", &log_level_, log_levels, IM_ARRAYSIZE(log_levels))) {
        //traact::util::initLogging(static_cast<spdlog::level::level_enum>(log_level_));
        SPDLOG_INFO("set log level {0}", log_level_);
        spdlog::set_level(static_cast<spdlog::level::level_enum>(log_level_));
    }

    if(dataflow_state_){
        drawDataflowState();
    }

    ImGui::End();


    if(facade_finished_.has_value() && attach_debug_renderer_){
        debug_renderer_->draw();
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

    if(attach_debug_renderer_){
        connectDebugRenderer();
    }




    std::thread start_thread([local_facade = facade_, this]() {
        local_facade->start();
        dataflow_state_ = local_facade->getDataflowState();
    });
    start_thread.detach();
}
void DebugRun::connectDebugRenderer() {
    debug_renderer_ = std::make_unique<DebugRenderer>();

    user_events_ = facade_->findComponents<component::SyncUserEventComponent>();

    debug_renderer_->init(*facade_, debug_sink_id_);

}
void DebugRun::setCurrentDataflow(std::shared_ptr<traact::gui::DataflowFile> dataflow) {
    auto isFacadeRunning = facade_ && facade_->isRunning();
    if(!dataflow || isFacadeRunning){
        return;
    }
    current_dataflow_ = dataflow;
}
bool DebugRun::canStart() {
    return current_dataflow_ != nullptr && !facade_finished_.has_value();
}
bool DebugRun::canStop() {
    return facade_ != nullptr && facade_finished_.has_value();
}
void DebugRun::parameterChanged(const std::string &instance_id) {
    if(facade_){

        facade_->parameterChanged(instance_id);
    }

}
void DebugRun::drawDataflowState() {
    ImGui::BeginChild("State");

    for(auto& event : user_events_){
        if(ImGui::Button(event->getName().c_str())) {
            event->fireEvent(0);
        }
    }


    ImGui::EndChild();

}
} // traact