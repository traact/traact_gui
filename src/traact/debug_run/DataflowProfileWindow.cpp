/** Copyright (C) 2022  Frieder Pankratz <frieder.pankratz@gmail.com> **/

#include "traact/imgui_util.h"
#include "DataflowProfileWindow.h"
#include "external/imgui-node-editor/utilities/drawing.h"
#include "external/imgui-node-editor/utilities/widgets.h"
#include <traact/util/Logging.h>

#include <utility>

namespace ed = ax::NodeEditor;

namespace traact::gui {

static ImColor getColorState(const dataflow::ProfileTimeResult &result) {
    switch (result.current_state) {

        case dataflow::IDLE: {
            return ImColor(255, 255, 0, 255);
        }
        case dataflow::STARTED: {
            return ImColor(0, 255, 0, 255);
        }
        case dataflow::FINISHED: {
            return ImColor(0, 0, 255, 255);
        }
        case dataflow::INVALID:
        default: {
            return ImColor(255, 0, 0, 255);
        }
    }

}

gui::DataflowProfileWindow::DataflowProfileWindow(const std::string &window_name, DebugRenderer *renderer)
    : DebugRenderComponent(100, 0, "invalid", window_name, renderer) {
    render_command_ = [this]() {
        draw();
    };
    context_ = ed::CreateEditor();
    task_type_visible_.fill(true);
}
void DataflowProfileWindow::init(dataflow::DataflowState::SharedPtr dataflow_state) {
    std::scoped_lock guard(dataflow_mutex_);
    dataflow_state_ = std::move(dataflow_state);

}
void DataflowProfileWindow::initGraph() {
    node_elements_.resize(dataflow_state_->getTimeDomainCount());
    link_elements_.resize(dataflow_state_->getTimeDomainCount());

    for (int i = 0; i < dataflow_state_->getTimeDomainCount(); ++i) {
        auto &processing =
            processing_.emplace_back(std::make_shared<dataflow::TimeDomainStateProcessing>(dataflow_state_->getState(i)));
        node_elements_[i].reserve(processing->getInfo().size());
        for (const auto &info : processing->getInfo()) {
            auto &node = node_elements_[i].emplace_back(info);
        }
    }

    for (int time_domain = 0; time_domain < dataflow_state_->getTimeDomainCount(); ++time_domain) {
        auto &link_elements = link_elements_[time_domain];
        auto &node_elements = node_elements_[time_domain];
        const auto &info = processing_[time_domain]->getInfo();

        for (size_t task_index = 0; task_index < info.size(); ++task_index) {
            for (auto pred_task_index : info[task_index].predecessors) {
                link_elements.emplace_back(node_elements.at(pred_task_index).producer_pin,
                                           node_elements.at(task_index).consumer_pin);
            }
        }
    }

    initialized_.store(true);
}
void DataflowProfileWindow::update(buffer::ComponentBuffer &buffer,
                                   std::vector<RenderCommand> &additional_commands) {

//    if (!initialized_.load()) {
//        return;
//    }
//    for (const auto &time_domain : processing_) {
//        time_domain->update(std::chrono::seconds(5));
//    }

}
void DataflowProfileWindow::draw() {

    for (const auto &time_domain : processing_) {
        time_domain->update(std::chrono::seconds(5));
    }

    ImGui::Begin(window_name_.c_str(), nullptr);
    ed::SetCurrentEditor(context_);
    ed::Begin("profile_editor");

    if (!initialized_.load()) {
        std::scoped_lock guard(dataflow_mutex_);
        if (dataflow_state_) {
            initGraph();
            show_details_ = false;
            setOnlyUserVisible();
            updateVisibility();
            drawGraph();
            layoutNodes();
        }
    } else {
        drawGraph();
    }

    ed::Suspend();
    if (ed::ShowBackgroundContextMenu()) {
        ImGui::OpenPopup("Profile Context Menu");
    }

    if (ImGui::BeginPopup("Profile Context Menu")) {

        if (ImGui::Checkbox("Time Step details", &show_details_)) {
            updateVisibility();
            layoutNodes();
        }
        if (ImGui::BeginMenu("Show")) {
            if (ImGui::MenuItem("all")) {
                setAllVisible();
                layoutNodes();
            }
            if (ImGui::MenuItem("only user")) {
                setOnlyUserVisible();
                layoutNodes();
            }
            for (int i = 0; i < (int) dataflow::task_util::TaskType::COUNT; ++i) {
                auto task_type = (dataflow::task_util::TaskType) i;
                bool type_visible = task_type_visible_[i];
                if (ImGui::Checkbox(dataflow::task_util::taskTypeEnumToName(task_type), &type_visible)) {
                    task_type_visible_[i] = type_visible;
                    updateVisibility();
                    layoutNodes();
                }
            }
            ImGui::EndMenu();
        }

        if (ImGui::MenuItem("Layout Nodes")) {
            layoutNodes();
        }

        ImGui::EndPopup();
    }

    ed::Resume();
    ed::End();

    ImGui::End();

}
void DataflowProfileWindow::drawGraph() {
    for (size_t time_domain_index = 0; time_domain_index < dataflow_state_->getTimeDomainCount();
         ++time_domain_index) {
        drawTimeDomain(time_domain_index);
    }
}
void DataflowProfileWindow::drawTimeDomain(size_t time_domain_index) {
    auto time_step_count =processing_[time_domain_index]->getTimeStepCount();
    const auto &profile = processing_[time_domain_index]->getProfileResult();
    auto &node_elements = node_elements_[time_domain_index];
    auto &link_elements = link_elements_[time_domain_index];

    const auto &task_profiles = profile.tasks;

    const static ImColor input_color(255, 0, 0, 255);
    const static ImColor output_color(0, 255, 0, 255);
    const static ImColor link_color(0, 0, 255, 255);
    const float TEXT_BASE_WIDTH = ImGui::CalcTextSize("A").x;
    const float TEXT_BASE_HEIGHT = ImGui::GetTextLineHeightWithSpacing();

    const float pin_size(20);
    const ImVec2 pin_vec(pin_size, pin_size);
    const ImVec2 half_pin_vec(pin_size / 2, pin_size / 2);
    ImVec2 group_size(200, 200);

    //SPDLOG_ERROR("profile result {0} {1}", profile.nano_seconds_per_frame.time, profile.nano_seconds_per_frame.std_dev);
    for (const auto &node : node_elements) {

        if (!node.visible) {
            continue;
        }


        const auto &task_profile = task_profiles[node.task_info.task_index];
        auto type_instance_id = fmt::format("{0}_{1}", dataflow::task_util::taskTypeEnumToName(node.task_info.task_type), node.task_info.pattern_instance_id);
        const auto &time_step_profile =  profile.time_step_tasks.at(type_instance_id);

        ed::BeginNode(node.node_id);
        ImGui::BeginGroup();
        if(show_details_) {
            ImGui::Text("%s", node.task_info.task_id.c_str());
            ImGui::Text("Event: %zu Mea Count: %zu ", task_profile->current_event, task_profile->measurement_count);
            ImGui::Text("Mean : %s \xC2\xB1%s", formatDuration(task_profile->mean_duration).c_str(),
                        formatDuration(task_profile->std_dev).c_str());
            ImGui::Text("Events/Second: %s", fmt::format("{0:3.2f}", task_profile->events_per_second).c_str());

        } else {
            ImGui::Text("%s", node.task_info.pattern_instance_id.c_str());
            ImGui::Text("Event: %zu Mea Count: %zu ", time_step_profile->current_event, time_step_profile->measurement_count);
            ImGui::Text("Mean : %s \xC2\xB1%s", formatDuration(time_step_profile->mean_duration).c_str(),
                        formatDuration(time_step_profile->std_dev).c_str());
            ImGui::Text("Events/Second: %s", fmt::format("{0:3.2f}", time_step_profile->events_per_second).c_str());

        }
        ImGui::EndGroup();

        if (ImGui::IsItemHovered()) {
            ImGui::SetNextWindowPos(ed::CanvasToScreen(ImGui::GetIO().MousePos)+ImVec2(20,0));

            ImGui::BeginTooltip();
            ImGui::Text("Type: %s", dataflow::task_util::taskTypeEnumToName(node.task_info.task_type));

            if(show_details_){

            } else {

            }

            ImGui::EndTooltip();
        }

        ImGui::BeginGroup();

        ImRect inputsRect;

        if (node.has_input) {
            ed::BeginPin(node.consumer_pin, ed::PinKind::Input);
            ax::Widgets::Icon(pin_vec, ax::Drawing::IconType::Circle, false, input_color, ImColor(32, 32, 32, 0));
            inputsRect = ImGui_GetItemRect();
            ed::PinPivotRect(inputsRect.GetTL() + half_pin_vec, inputsRect.GetTL() + half_pin_vec);
            ed::EndPin();
        }
        ImGui::EndGroup();


        ImGui::SameLine();

        ImGui::BeginGroup();
        float offset = TEXT_BASE_WIDTH * static_cast<float>(node.name.length());
        ed::BeginPin(node.producer_pin, ed::PinKind::Output);
        ImGui::SameLine(offset);
        ax::Widgets::Icon(pin_vec, ax::Drawing::IconType::Circle, false, output_color, ImColor(32, 32, 32, 0));
        inputsRect = ImGui_GetItemRect();
        ed::PinPivotRect(inputsRect.GetTL() + half_pin_vec, inputsRect.GetTL() + half_pin_vec);
        ed::EndPin();
        ImGui::EndGroup();

        if(!show_details_){
            for (int i = 1; i < time_step_count; ++i) {
                ImGui::Text(" ");
            }
        }





        ed::EndNode();

        auto drawList = ed::GetNodeBackgroundDrawList(node.node_id);
        auto node_pos = ed::GetNodePosition(node.node_id);
        auto node_size = ed::GetNodeSize(node.node_id);
        if (show_details_) {
            drawList->AddRect(node_pos + ImVec2(2, 2),
                              node_pos + node_size - ImVec2(2, 2),
                              getColorState(*task_profile),
                              ed::GetStyle().NodeRounding);
        } else {
            float node_width = node_size.y;

            ImVec2 time_step_size(node_size.x, TEXT_BASE_HEIGHT);
            ImVec2 padding(2,2);
            for (int i = 0; i < time_step_count; ++i) {
                ImVec2 time_step_pos(node_pos.x, node_pos.y+(4.0f + i)*TEXT_BASE_HEIGHT+pin_size/2);
                drawList->AddRect(time_step_pos + padding,
                                  time_step_pos + time_step_size - padding,
                                  getColorState(*time_step_profile->time_steps[i]),
                                  ed::GetStyle().NodeRounding);
            }
        }

    }

    for (auto &link : link_elements) {
        if (!link.visible) {
            continue;
        }
        ed::Link(link.link_id, link.start_pin, link.end_pin, link_color, 2.0f);
    }

}
void DataflowProfileWindow::layoutNodes() {

    ImVec2 start_position(0, 0);
    for (const auto &time_domain_nodes : node_elements_) {
        start_position = layoutTimeDomain2(time_domain_nodes, start_position);
    }
}
ImVec2 DataflowProfileWindow::layoutTimeDomain(const std::vector<NodeElement> &time_domain_nodes,
                                               ImVec2 start_position) {
    std::map<size_t, std::map<std::string, size_t>> time_step_to_pattern_to_index;
    for (int i = 0; i < time_domain_nodes.size(); ++i) {
        const auto &node = time_domain_nodes[i];
        auto &pattern_to_index = time_step_to_pattern_to_index[node.task_info.time_step];
        pattern_to_index[fmt::format("{0}_{1}",
                                     dataflow::task_util::taskTypeEnumToName(node.task_info.task_type),
                                     node.task_info.pattern_instance_id)] = i;
    }

    std::vector<std::vector<std::string>>
        node_table = layoutTimeStep(time_domain_nodes, time_step_to_pattern_to_index.at(0));

    ImVec2 node_position(0, 0);
    float padding = 10;
    float node_distance = 50;
    float total_height = 0;
    for (const auto &column : node_table) {
        float max_width = 0;
        for (const auto &cell : column) {
            for (size_t time_step_index = 0; time_step_index < time_step_to_pattern_to_index.size();
                 ++time_step_index) {
                SPDLOG_INFO("layout node name {0} time step {1}", cell, time_step_index);
                auto node_index = time_step_to_pattern_to_index[time_step_index].find(cell);
                if (node_index == time_step_to_pattern_to_index[time_step_index].end()) {
                    continue;
                }
                SPDLOG_INFO("set node position node name {0} time step {1}, {2} {3}",
                            cell,
                            time_step_index,
                            node_position.x,
                            node_position.y);
                const auto &node = time_domain_nodes[node_index->second];
                ed::SetNodePosition(node.node_id, node_position);
                auto node_size = ed::GetNodeSize(node.node_id);
                max_width = std::max(max_width, node_size.x);
                node_position.y += node_size.y + padding;
            }
            node_position.y += node_distance;

        }
        node_position.x += max_width + node_distance;
        total_height = std::max(total_height, node_position.y);
        node_position.y = 0;

    }

    return ImVec2(0, total_height);

}
ImVec2 DataflowProfileWindow::layoutTimeDomain2(const std::vector<NodeElement> &time_domain_nodes,
                                                ImVec2 start_position) {

    struct Cell {
        using PositionFunc = std::function<ImVec2()>;
        Cell(NodeElement const *node) : node(node), position{} {};
        Cell(NodeElement const *node, PositionFunc get_position) : node(node), position(std::move(get_position)) {};
        Cell(const Cell &) = default;
        Cell(Cell &&) = default;
        ~Cell() = default;

        NodeElement const *node;
        PositionFunc position;
        bool has_grid_position{false};
    };



    // sort by time step, 0 is main
    std::map<size_t, std::vector<NodeElement const *>> time_step_to_node;
    for (const auto &node : time_domain_nodes) {
        auto &nodes = time_step_to_node[node.task_info.time_step];
        nodes.emplace_back(&node);
    }

    auto time_step_count = time_step_to_node.size();

    float time_step_distance = static_cast<float>(time_step_count);
    float column_distance = 1.0f;
    if (!show_details_) {
        time_step_distance = 1.0f;
        if (task_type_visible_[static_cast<int>(dataflow::task_util::TaskType::SEAM_ENTRY)]) {
            time_step_distance += 1.0f;
        }
        column_distance = 0.0f;
    }

    std::vector<Cell> cells;
    cells.reserve(time_domain_nodes.size());
    // add time step 0 to all cells
    for (NodeElement const *node : time_step_to_node[0]) {
        cells.emplace_back(node);
    }

    ImVec2 max_node_size{0, 0};
    for (const auto &cell : cells) {
        ImVec2 padding{10, 10};
        auto node_size = ed::GetNodeSize(cell.node->node_id);
        max_node_size.x = std::max(max_node_size.x, node_size.x);
        max_node_size.y = std::max(max_node_size.y, node_size.y);
    }

    auto findCell = [&cells](const std::string &task_id) -> std::optional<Cell *> {
        auto result = std::find_if(cells.begin(), cells.end(), [&task_id](const auto &item) {
            return item.node->task_info.task_id == task_id;
        });
        if (result == cells.end()) {
            return {};
        } else {
            return &(*result);
        }
    };

    auto relative_to =
        [](Cell *parent, Cell *self, ImVec2 direction, ImVec2 padding = ImVec2(10, 10)) -> Cell::PositionFunc {
            return [parent, self, direction, padding]() {
                ImVec2 pos = parent->position();
                ImVec2 node_size;
                if (direction.x > 0) {
                    node_size = ed::GetNodeSize(parent->node->node_id);
                } else {
                    node_size = ed::GetNodeSize(self->node->node_id);
                }
                return pos + (node_size + padding) * direction;
            };
        };

    auto left_bottom_of = [](Cell *parent, Cell *self, ImVec2 offset = ImVec2(-10, 10)) -> Cell::PositionFunc {
        return [parent, self, offset]() {
            auto pos = parent->position();
            auto node_size = ed::GetNodeSize(self->node->node_id);
            return pos - node_size + offset;
        };
    };

    auto left_top_of = [](Cell *parent, Cell *self, ImVec2 offset = ImVec2(-10, 10)) -> Cell::PositionFunc {
        return [parent, self, offset]() {
            auto pos = parent->position();
            auto node_size = ed::GetNodeSize(self->node->node_id);
            return pos - node_size + offset;
        };
    };

    auto right_bottom_of = [](Cell *parent, ImVec2 offset = ImVec2(10, 10)) -> Cell::PositionFunc {
        return [parent, offset]() {
            auto pos = parent->position();
            auto node_size = ed::GetNodeSize(parent->node->node_id);
            return pos + node_size + offset;
        };
    };

    auto right_top_of = [](Cell *parent, ImVec2 offset = ImVec2(10, -10)) -> Cell::PositionFunc {
        return [parent, offset]() {
            auto pos = parent->position();
            auto node_size = ed::GetNodeSize(parent->node->node_id);
            return ImVec2(pos.x + node_size.x, pos.y - node_size.y) + offset;
        };
    };

    auto bottom_of = [time_step_distance](Cell *parent, ImVec2 offset = ImVec2(0, 0)) -> Cell::PositionFunc {
        return [parent, time_step_distance, offset]() {
            auto pos = parent->position();
            auto node_size = ed::GetNodeSize(parent->node->node_id);
            return pos + ImVec2{0, (node_size.y + 10) * time_step_distance} + offset;
        };
    };

    auto next_column_of =
        [time_step_distance, max_node_size, column_distance](Cell *parent,
                                                             ImVec2 offset = ImVec2(30, 0)) -> Cell::PositionFunc {
            return [parent, time_step_distance, max_node_size, offset, column_distance]() {
                auto pos = parent->position();
                //auto node_size = ed::GetNodeSize(parent->node->node_id);
                return pos + ImVec2{(max_node_size.x) * (time_step_distance + column_distance), 0} + offset;
            };
        };

    auto sub_node_of = [](Cell *parent, ImVec2 offset = ImVec2(30, 10)) -> Cell::PositionFunc {
        return [parent, offset]() {
            auto pos = parent->position();
            auto node_size = ed::GetNodeSize(parent->node->node_id);
            return ImVec2(pos.x + node_size.x, pos.y + node_size.y) + offset;
        };
    };

    const std::string globalStart =
        dataflow::task_util::getTaskId(dataflow::task_util::kSeamStart, dataflow::task_util::kGlobalStart, 0);
    // add start position
    {

        const std::string globalEntry =
            dataflow::task_util::getTaskId(dataflow::task_util::kSeamEntry, dataflow::task_util::kGlobalStart, 0);

        const std::string globalEnd =
            dataflow::task_util::getTaskId(dataflow::task_util::kSeamEnd, dataflow::task_util::kGlobalStart, 0);

        auto entry_cell = findCell(globalEntry);
        if (entry_cell) {
            entry_cell.value()->position = []() -> ImVec2 {
                return {0, 0};
            };
        } else {
            SPDLOG_ERROR("global entry not found");
            return start_position;
        }

        auto start_cell = findCell(globalStart);
        if (start_cell) {
            start_cell.value()->position = right_bottom_of(entry_cell.value());
        } else {
            SPDLOG_ERROR("global start not found");
            return start_position;
        }

        auto end_cell = findCell(globalEnd);
        if (end_cell) {
            end_cell.value()->position = right_top_of(start_cell.value());
        } else {
            SPDLOG_ERROR("global start not found");
            return start_position;
        }
    }

    std::vector<std::vector<Cell *>> table;
    // layout time step 0
    {
        auto all_predecessors_in_table = [&table, &time_domain_nodes](NodeElement const *node) -> bool {

            for (const auto &predecessor_index : node->task_info.predecessors) {
                auto &predecessor = time_domain_nodes[predecessor_index];

                switch (predecessor.task_info.task_type) {
                    case dataflow::task_util::TaskType::SOURCE:
                    case dataflow::task_util::TaskType::COMPONENT:
                    case dataflow::task_util::TaskType::MODULE:
                    case dataflow::task_util::TaskType::CUDA_COMPONENT:
                    case dataflow::task_util::TaskType::CUDA_FLOW: {

                        break;
                    }

                    default: {
                        // ignore non-user component tasks for layout
                        continue;
                    }
                }

                bool predecessor_found{false};
                for (const auto &column : table) {
                    auto predecessor_cell = std::find_if(column.begin(), column.end(), [predecessor](const Cell *item) {
                        return predecessor.node_id == item->node->node_id;
                    });
                    if (predecessor_cell != column.cend()) {
                        predecessor_found = true;
                        break;
                    }
                }
                if (!predecessor_found) {
                    return false;
                }

            }
            return true;
        };

        bool layout_finished{false};
        while (!layout_finished) {

            layout_finished = true;
            std::vector<Cell *> column;
            for (auto &cell : cells) {

                switch (cell.node->task_info.task_type) {
                    case dataflow::task_util::TaskType::SOURCE:
                    case dataflow::task_util::TaskType::COMPONENT:
                    case dataflow::task_util::TaskType::MODULE:
                    case dataflow::task_util::TaskType::CUDA_COMPONENT:
                    case dataflow::task_util::TaskType::CUDA_FLOW: {
                        break;
                    }

                    default: {
                        // ignore non-user component tasks for layout
                        continue;
                    }
                }

                if (!cell.has_grid_position) {
                    if (all_predecessors_in_table(cell.node)) {
                        cell.has_grid_position = true;
                        column.emplace_back(&cell);
                    } else {
                        layout_finished = false;

                    }
                }
            }

            if (!column.empty()) {
                table.emplace_back(column);
            } else {
                SPDLOG_ERROR("new column in layout has 0 elements, endless loop, abort");
                return start_position;
            }
        }

        auto start_cell = findCell(globalStart);
        if (!start_cell) {
            SPDLOG_ERROR("start cell not found");
            return start_position;
        }
        if (table.empty() || table.begin()->empty()) {
            SPDLOG_ERROR("component table is empty");
            return start_position;
        }

        Cell *column_start{nullptr};
        for (size_t column_index = 0; column_index < table.size(); ++column_index) {
            auto &column = table[column_index];
            if (column_index == 0) {
                column_start = column.front();
                column_start->position = right_bottom_of(start_cell.value(), ImVec2(100, 100));
            } else {
                Cell *prev_start = column_start;
                column_start = column.front();
                column_start->position = next_column_of(prev_start);
            }

            for (size_t row_index = 1; row_index < column.size(); ++row_index) {
                auto *cell = column[row_index];
                cell->position = bottom_of(column[row_index - 1]);
            }
        }

    }

    // add other time step tasks as subtasks of the previous time step
    auto find_cell_time_step = [&cells](NodeElement const *node, int search_time_step) -> std::optional<Cell *> {
        auto result = std::find_if(cells.begin(), cells.end(), [&node, search_time_step](const auto &item) {
            return item.node->task_info.pattern_instance_id == node->task_info.pattern_instance_id &&
                item.node->task_info.time_step == search_time_step;
        });
        if (result == cells.end()) {
            return {};
        } else {
            return {&(*result)};
        }
    };
    for (int i = 1; i < time_domain_nodes.size(); ++i) {
        for (NodeElement const *node : time_step_to_node[i]) {
            auto prev_cell = find_cell_time_step(node, i - 1);
            if (prev_cell) {
                cells.emplace_back(node, sub_node_of(prev_cell.value()));
            } else {
                SPDLOG_ERROR("previous cell not found  {0}", node->task_info.task_id);
            }
        }
    }

    // set time step start and end tasks using the component table
    {
        if (table.empty() || table.front().empty()) {
            SPDLOG_ERROR("no component in table");
            return start_position;
        }
        auto *top_left = table.front()[0];
        auto *top_right = table.back()[0];
        auto ts_0_start = findCell(dataflow::task_util::getTaskId(dataflow::task_util::kControlFlow,
                                                                  dataflow::task_util::kTimeStepStart,
                                                                  0));
        auto ts_0_end = findCell(dataflow::task_util::getTaskId(dataflow::task_util::kControlFlow,
                                                                dataflow::task_util::kTimeStepEnd,
                                                                0));

        if (!ts_0_start || !ts_0_end) {
            SPDLOG_ERROR("start/end task not found");
            return start_position;
        }
        Cell *start_cell = ts_0_start.value();
        Cell *end_cell = ts_0_end.value();

        start_cell->position = relative_to(top_left, start_cell, ImVec2(-time_step_distance * 1.5, 0));
        end_cell->position = relative_to(top_right, end_cell, ImVec2(time_step_distance * 1.5, 0));
        for (int i = 1; i < time_step_count; ++i) {
            auto ts_n_start = findCell(dataflow::task_util::getTaskId(dataflow::task_util::kControlFlow,
                                                                      dataflow::task_util::kTimeStepStart,
                                                                      i));
            auto ts_n_end = findCell(dataflow::task_util::getTaskId(dataflow::task_util::kControlFlow,
                                                                    dataflow::task_util::kTimeStepEnd,
                                                                    i));
            if (!ts_n_start || !ts_n_end) {
                SPDLOG_ERROR("start/end task not found");
                return start_position;
            }
            ts_n_start.value()->position = relative_to(start_cell, ts_n_start.value(), ImVec2(0, time_step_distance));
            ts_n_end.value()->position = relative_to(end_cell, ts_n_end.value(), ImVec2(0, time_step_distance));
            start_cell = ts_n_start.value();
            end_cell = ts_n_end.value();
        }

    }


    // set seam positions
    {
        auto find_cell_seam = [&cells](NodeElement const *node, const char *seam) -> std::optional<Cell *> {
            std::string seam_name = dataflow::task_util::getTaskId(seam,
                                                                   node->task_info.pattern_instance_id.c_str(),
                                                                   node->task_info.time_step);
            auto result = std::find_if(cells.begin(), cells.end(), [seam_name](const auto &item) {
                return item.node->task_info.task_id == seam_name;
            });
            if (result == cells.end()) {
                return {};
            } else {
                return {&(*result)};
            }
        };

        for (auto &cell : cells) {
            if (cell.node->task_info.time_step != 0) {
                continue;
            }

            // all non seam tasks
            switch (cell.node->task_info.task_type) {
                case dataflow::task_util::TaskType::SEAM_ENTRY:
                case dataflow::task_util::TaskType::SEAM_START:
                case dataflow::task_util::TaskType::SEAM_END: {
                    continue;
                }

                default: {
                    break;
                }
            }

            auto seam_entry = find_cell_seam(cell.node, dataflow::task_util::kSeamEntry);
            auto seam_start = find_cell_seam(cell.node, dataflow::task_util::kSeamStart);
            auto seam_end = find_cell_seam(cell.node, dataflow::task_util::kSeamEnd);
            if (!seam_entry || !seam_start || !seam_end) {
                continue;
            }

            seam_start.value()->position = relative_to(&cell, seam_start.value(), ImVec2(-1, 0));
            seam_entry.value()->position =
                relative_to(seam_start.value(), seam_entry.value(), ImVec2(0, time_step_distance - 1));
            seam_end.value()->position = relative_to(seam_entry.value(), seam_end.value(), ImVec2(1, 0));

        }

        for (int i = 0; i < time_step_count; ++i) {
            auto ts_n_start = findCell(dataflow::task_util::getTaskId(dataflow::task_util::kControlFlow,
                                                                      dataflow::task_util::kTimeStepStart,
                                                                      i));
            auto ts_n_self_entry = findCell(dataflow::task_util::getTaskId(dataflow::task_util::kSeamEntry,
                                                                           dataflow::task_util::kTimeStepSelf,
                                                                           i));
            auto ts_n_self_start = findCell(dataflow::task_util::getTaskId(dataflow::task_util::kSeamStart,
                                                                           dataflow::task_util::kTimeStepSelf,
                                                                           i));
            auto ts_n_self_end = findCell(dataflow::task_util::getTaskId(dataflow::task_util::kSeamEnd,
                                                                         dataflow::task_util::kTimeStepSelf,
                                                                         i));
            if (!ts_n_start || !ts_n_self_entry || !ts_n_self_start || !ts_n_self_end) {
                SPDLOG_ERROR("start/end task not found");
                return start_position;
            }
            ts_n_self_start.value()->position =
                relative_to(ts_n_start.value(), ts_n_self_start.value(), ImVec2(-1, -1));
            ts_n_self_entry.value()->position =
                relative_to(ts_n_self_start.value(), ts_n_self_entry.value(), ImVec2(-1, 0));
            ts_n_self_end.value()->position = relative_to(ts_n_self_entry.value(), ts_n_self_end.value(), ImVec2(0, 1));
        }
    }

    std::array<int, static_cast<int>(dataflow::task_util::TaskType::COUNT)> task_count;
    task_count.fill(0);
    for (auto &cell : cells) {
        if (!cell.position) {
            SPDLOG_ERROR("missing layout for task {0}", cell.node->task_info.task_id);
            int column = static_cast<int>(cell.node->task_info.task_type);
            cell.position = [count = task_count[column]++, column, max_node_size]() {
                return ImVec2(max_node_size.x * (column - 10), max_node_size.y * count);
            };
        }
    }

    float total_height{0};
    for (const auto &cell : cells) {
        auto position = cell.position();
        ed::SetNodePosition(cell.node->node_id, position);
        total_height = std::max(total_height, position.y);
    }

    return ImVec2(0, total_height);
}
std::vector<std::vector<std::string>> DataflowProfileWindow::layoutTimeStep(const std::vector<NodeElement> &all_nodes,
                                                                            std::map<std::string,
                                                                                     size_t> &time_step_nodes) {

    std::vector<std::vector<std::string>> node_table;

    auto node_count = time_step_nodes.size();

    std::vector<std::string> unsorted_nodes;
    unsorted_nodes.reserve(node_count);

    // sort out sources and not connected components for the first two columns
    // 0: source (no input), rest unsorted
//    node_table.resize(3);
//    for(const auto& name_index : time_step_nodes) {
//        const auto &node = all_nodes[name_index.second];
//
//        if (node.task_info.task_type == dataflow::task_util::TaskType::SOURCE) {
//            SPDLOG_INFO("add source flow {0}", node.task_info.task_id);
//            node_table[2].emplace_back(name_index.first);
//        } else if (node.task_info.task_type == dataflow::task_util::TaskType::SEAM_ENTRY
//            || node.task_info.task_type == dataflow::task_util::TaskType::SEAM_START
//            || node.task_info.task_type == dataflow::task_util::TaskType::SEAM_END
//            || node.task_info.task_type == dataflow::task_util::TaskType::CONTROL_FLOW) {
//            SPDLOG_INFO("add control flow {0}", node.task_info.task_id);
//            node_table[1].emplace_back(name_index.first);
//
//        } else if (node.task_info.task_id == "start_entry") {
//            SPDLOG_INFO("add start task {0}", node.task_info.task_id);
//            node_table[0].emplace_back(name_index.first);
//        } else {
//            SPDLOG_INFO("add unsorted {0} {1}",
//                        node.task_info.task_id,
//                        dataflow::task_util::taskTypeEnumToName(node.task_info.task_type));
//            unsorted_nodes.emplace_back(name_index.first);
//        }
//
//    }

    auto sort_by_output = [](const std::string &a, const std::string &b) -> bool {
        return a > b;
    };

    //sort(node_table[0].begin(), node_table[0].end(), sort_by_output);
    //sort(node_table[1].begin(), node_table[1].end(), sort_by_output);

    auto isInTable =
        [&node_table, &unsorted_nodes](const NodeElement *current_node, const NodeElement *test_node) -> bool {

            if (current_node->task_info.time_step != test_node->task_info.time_step) {
                return true;
            }
            if (current_node->task_info.task_type != dataflow::task_util::TaskType::COMPONENT) {
                return true;
            }

            std::string test_node_name = fmt::format("{0}_{1}",
                                                     dataflow::task_util::taskTypeEnumToName(test_node->task_info.task_type),
                                                     test_node->task_info.pattern_instance_id);

            for (const auto &column : node_table) {
                auto result =
                    std::find_if(column.cbegin(), column.cend(), [&test_node_name](const std::string &b) -> bool {
                        return test_node_name == b;
                    });
                if (result != column.cend()) {
                    return true;
                };
            }

            return false;
//        auto result = std::find(unsorted_nodes.begin(), unsorted_nodes.end(), test_node_name);
//        return result == unsorted_nodes.end();
//        return false;
        };

    while (!unsorted_nodes.empty()) {

        std::vector<std::string> next_column;
        std::vector<std::string> current_nodes;
        current_nodes.reserve(node_count);
        for (const auto &node : unsorted_nodes) {
            current_nodes.emplace_back(node);
        }

        for (const auto &node_name : current_nodes) {
            const auto &node = all_nodes[time_step_nodes.at(node_name)];
            bool only_connected_to_prev = true;

            for (const auto &task_index : node.task_info.predecessors) {
                SPDLOG_INFO("test node {0}", all_nodes[task_index].task_info.task_id);
                if (!isInTable(&node, &all_nodes[task_index])) {
                    only_connected_to_prev = false;
                    break;
                }
            }
            if (only_connected_to_prev) {
                next_column.push_back(node_name);
                auto remove_it = std::remove(unsorted_nodes.begin(), unsorted_nodes.end(), node_name);
                unsorted_nodes.erase(remove_it, unsorted_nodes.end());
            }
        }

        sort(next_column.begin(), next_column.end(), sort_by_output);
        node_table.emplace_back(next_column);
    }

    return node_table;
}
void DataflowProfileWindow::updateVisibility() {

    for (int time_domain = 0; time_domain < node_elements_.size(); ++time_domain) {
        auto &time_domain_nodes = node_elements_[time_domain];
        auto &time_domain_link = link_elements_[time_domain];
        for (auto &node : time_domain_nodes) {
            auto type_index = static_cast<int>(node.task_info.task_type);
            node.visible = task_type_visible_[type_index];
            if (!show_details_) {
                node.visible = node.visible && node.task_info.time_step == 0;
            }

        }

        for (auto &link : time_domain_link) {
            auto start_node = std::find_if(time_domain_nodes.begin(),
                                           time_domain_nodes.end(),
                                           [start_pin = link.start_pin](const NodeElement &item) {
                                               return item.producer_pin == start_pin;
                                           });
            auto end_node = std::find_if(time_domain_nodes.begin(),
                                         time_domain_nodes.end(),
                                         [end_pin = link.end_pin](const NodeElement &item) {
                                             return item.consumer_pin == end_pin;
                                         });
            if (start_node != time_domain_nodes.end() && end_node != time_domain_nodes.end()) {
                link.visible = start_node->visible && end_node->visible;
            } else {
                link.visible = false;
                SPDLOG_ERROR("could not find start/end nodes of link");
            }
        }
    }
}
void DataflowProfileWindow::setAllVisible() {
    task_type_visible_.fill(true);
    updateVisibility();

}
void DataflowProfileWindow::setOnlyUserVisible() {
    task_type_visible_.fill(false);
    task_type_visible_[static_cast<size_t>(dataflow::task_util::TaskType::SOURCE)] = true;
    task_type_visible_[static_cast<size_t>(dataflow::task_util::TaskType::COMPONENT)] = true;
    task_type_visible_[static_cast<size_t>(dataflow::task_util::TaskType::MODULE)] = true;
    task_type_visible_[static_cast<size_t>(dataflow::task_util::TaskType::CUDA_COMPONENT)] = true;
    task_type_visible_[static_cast<size_t>(dataflow::task_util::TaskType::CUDA_FLOW)] = true;
    updateVisibility();
}

gui::DataflowProfileWindow::NodeElement::NodeElement(const dataflow::ProcessingTaskInfo &task_info) : task_info(
    task_info),
                                                                                                      node_id(editor::utils::GetNextId()),
                                                                                                      producer_pin(
                                                                                                          editor::utils::GetNextId()),
                                                                                                      consumer_pin(
                                                                                                          editor::utils::GetNextId()) {

    name = fmt::format("{0} : {1}", task_info.pattern_instance_id, task_info.time_step);
    has_input = !task_info.predecessors.empty();
}

gui::DataflowProfileWindow::LinkElement::LinkElement(const ax::NodeEditor::PinId &start_pin,
                                                     const ax::NodeEditor::PinId &end_pin)
    : link_id(editor::utils::GetNextId()), start_pin(start_pin), end_pin(end_pin) {}
} // traact