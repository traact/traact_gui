/** Copyright (C) 2022  Frieder Pankratz <frieder.pankratz@gmail.com> **/

#ifndef TRAACT_GUI_SRC_TRAACT_GUI_DEBUG_RUN_DEBUGRUN_H_
#define TRAACT_GUI_SRC_TRAACT_GUI_DEBUG_RUN_DEBUGRUN_H_

#include <memory>
#include "traact_gui/DataflowFile.h"
#include "DebugRenderer.h"

#include <traact/userEvent/component/SyncUserEventComponent.h>
namespace traact::gui {

class DebugRun {
 public:
    void draw();
    void parameterChanged(const std::string &instance_id);
    void setCurrentDataflow(std::shared_ptr<traact::gui::DataflowFile> dataflow);
 private:
    std::shared_ptr<traact::gui::DataflowFile> current_dataflow_;
    std::shared_ptr<traact::DefaultFacade> facade_;
    traact::dataflow::DataflowState::SharedPtr dataflow_state_;
    std::optional<std::shared_future<void>> facade_finished_;
    bool attach_debug_renderer_{true};
    std::string debug_sink_id_{"debug_sink"};

    std::unique_ptr<DebugRenderer> debug_renderer_;
    int log_level_{SPDLOG_LEVEL_INFO};
    std::vector<std::shared_ptr<component::SyncUserEventComponent>> user_events_;
    bool canStart();
    bool canStop();
    void startDataflow();
    void connectDebugRenderer();
    void drawDataflowState();
};

} // traact

#endif //TRAACT_GUI_SRC_TRAACT_GUI_DEBUG_RUN_DEBUGRUN_H_
