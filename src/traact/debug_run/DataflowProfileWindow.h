/** Copyright (C) 2022  Frieder Pankratz <frieder.pankratz@gmail.com> **/

#ifndef TRAACT_GUI_SRC_TRAACT_DEBUG_RUN_DATAFLOWPROFILEWINDOW_H_
#define TRAACT_GUI_SRC_TRAACT_DEBUG_RUN_DATAFLOWPROFILEWINDOW_H_


#include "DebugRenderComponent.h"
#include <traact/dataflow/state/TimeDomainStateResult.h>
#include <traact/dataflow/state/DataflowState.h>
#include "external/imgui-node-editor/imgui_node_editor.h"
#include "traact/editor/EditorUtils.h"
namespace traact::gui {


class DataflowProfileWindow : public DebugRenderComponent{
 public:
    DataflowProfileWindow(const std::string &window_name,
                   DebugRenderer *renderer);
    virtual ~DataflowProfileWindow() = default;

    void init(dataflow::DataflowState::SharedPtr dataflow_state);

    void update(buffer::ComponentBuffer &buffer, std::vector<RenderCommand> &additional_commands) override;

 private:
    struct NodeElement {
        explicit NodeElement(const dataflow::ProcessingTaskInfo &task_info);
        dataflow::ProcessingTaskInfo task_info;
        ax::NodeEditor::NodeId node_id;
        ax::NodeEditor::PinId consumer_pin;
        ax::NodeEditor::PinId producer_pin;
        std::string name;
        bool has_input;
        bool visible{true};
        //ImVec2 position;
    };
    struct LinkElement {
        LinkElement(const ax::NodeEditor::PinId &start_pin, const ax::NodeEditor::PinId &end_pin);
        ax::NodeEditor::LinkId link_id;
        ax::NodeEditor::PinId start_pin;
        ax::NodeEditor::PinId end_pin;
        bool visible{true};
    };
    std::mutex dataflow_mutex_;
    dataflow::DataflowState::SharedPtr dataflow_state_;
    std::vector<std::shared_ptr<dataflow::TimeDomainStateProcessing>> processing_;
    std::vector<std::vector<NodeElement>> node_elements_;
    std::vector<std::vector<LinkElement>> link_elements_;
    std::atomic_bool initialized_{false};
    std::array<bool, static_cast<size_t>(dataflow::task_util::TaskType::COUNT)> task_type_visible_;
    bool show_details_{true};

    ax::NodeEditor::EditorContext* context_ = nullptr;

    void draw();

    void drawTimeDomain(size_t time_domain_index);
    void layoutNodes();
    void initGraph();
    ImVec2 layoutTimeDomain(const std::vector<NodeElement> &time_domain_nodes,ImVec2 start_position);
    ImVec2 layoutTimeDomain2(const std::vector<NodeElement> &time_domain_nodes,ImVec2 start_position);
    void drawGraph();
    std::vector<std::vector<std::string>> layoutTimeStep(const std::vector<NodeElement> &all_nodes,
                                                         std::map<std::string, size_t> &time_step_nodes);
    void updateVisibility();
    void setAllVisible();
    void setOnlyUserVisible();
};

} // traact

#endif //TRAACT_GUI_SRC_TRAACT_DEBUG_RUN_DATAFLOWPROFILEWINDOW_H_
