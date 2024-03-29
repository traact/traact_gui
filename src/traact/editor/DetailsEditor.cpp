/** Copyright (C) 2022  Frieder Pankratz <frieder.pankratz@gmail.com> **/

#include "DetailsEditor.h"
#include "external/imgui_misc/imgui_stdlib.h"
namespace traact {
void gui::DetailsEditor::operator()(std::shared_ptr<DataflowFile> &dataflow_file) const {
    if (!dataflow_file) {
        return;
    }
    auto &graph_instance = dataflow_file->graph_editor_.Graph;
    ImGui::Text("%s", graph_instance->name.c_str());
    ImGui::SameLine();

    static std::string dataflow_name;
    if (ImGui::Button("Change name")) {
        ImGui::OpenPopup("change_dataflow_name_popup");
        dataflow_name = graph_instance->name;
    }
    if (ImGui::BeginPopup("change_dataflow_name_popup")) {
        ImGui::InputText(" new name", &dataflow_name);
        if (ImGui::IsItemDeactivatedAfterEdit()) {
            graph_instance->name = dataflow_name;
        }

        ImGui::EndPopup();
    }

    bool has_changed{false};

    for (auto &time_domain : graph_instance->timedomain_configs) {
        ImGui::Text("Time Domain %lu", time_domain.first);
        auto &time_domain_config = time_domain.second;
        double min_freq = 0;
        double max_freq = 100;
        ImGui::SliderScalar("Sensor frequency",
                            ImGuiDataType_Double,
                            &time_domain_config.sensor_frequency,
                            &min_freq,
                            &max_freq);
        has_changed |= ImGui::IsItemDeactivatedAfterEdit();
        ImGui::DragInt("Worker Count", &time_domain_config.cpu_count, 0.1f, -std::thread::hardware_concurrency(), 100);
        has_changed |= ImGui::IsItemDeactivatedAfterEdit();
        uint64_t min_buffer{1}, max_buffer{10};
        ImGui::SliderScalar("Buffer Count",
                            ImGuiDataType_U64,
                            &time_domain_config.ringbuffer_size,
                            &min_buffer,
                            &max_buffer);
        has_changed |= ImGui::IsItemDeactivatedAfterEdit();

        const char *source_event_names[] = {"WAIT_FOR_BUFFER", "IMMEDIATE_RETURN"};
        int source_event_index = static_cast<int>(time_domain_config.source_mode);

        if (ImGui::Combo("Source Mode", &source_event_index, source_event_names, IM_ARRAYSIZE(source_event_names))) {
            time_domain_config.source_mode = static_cast<SourceMode>(source_event_index);
            has_changed = true;
        }

        const char *missing_event_names[] = {"WAIT_FOR_EVENT", "CANCEL_OLDEST"};
        int missing_event_index = static_cast<int>(time_domain_config.missing_source_event_mode);
        if (ImGui::Combo("Missing Event Mode",
                         &missing_event_index,
                         missing_event_names,
                         IM_ARRAYSIZE(missing_event_names))) {
            time_domain_config.missing_source_event_mode = static_cast<MissingSourceEventMode>(missing_event_index);
            has_changed = true;
        }

        using namespace std::chrono;
        int64_t max_offset = duration_cast<milliseconds>(time_domain_config.max_offset).count();

        ImGui::InputScalar("Max offset (ms)", ImGuiDataType_S64, &max_offset);
        if (ImGui::IsItemDeactivatedAfterEdit()) {
            time_domain_config.max_offset = milliseconds(max_offset);
            has_changed = true;
        }
        int64_t max_delay = duration_cast<milliseconds>(time_domain_config.max_delay).count();
        ImGui::InputScalar("Max delay (ms)", ImGuiDataType_S64, &max_delay);
        if (ImGui::IsItemDeactivatedAfterEdit()) {
            time_domain_config.max_delay = milliseconds(max_delay);
            has_changed = true;
        }
        if(ImGui::Checkbox("Profile", &time_domain_config.profile)) {
            has_changed = true;
        }

        //time_domain_config.max_offset
        //time_domain_config.max_delay

    }

    if (has_changed && onChange) {
        onChange({dataflow_file});
    }

}
void gui::DetailsEditor::operator()(std::shared_ptr<traact::pattern::instance::PatternInstance> &pattern_instance) const {
    if (!pattern_instance) {
        return;
    }
    bool has_changed{false};
    ImGui::Text("%s", pattern_instance->instance_id.c_str());
    for (auto &[group_name, group_index] : pattern_instance->port_group_name_to_index) {
        ImGui::Text("%s", group_name.c_str());
        for (auto &group_instance : pattern_instance->port_groups[group_index]) {
            ImGui::Text("Group instance %d", group_instance->port_group_instance_index);
            for (auto &parameter : group_instance->port_group.parameter.items()) {
                const auto *parameter_c_str = parameter.key().c_str();
                auto &parameter_value = parameter.value()["value"];
                auto &parameter_default = parameter.value()["default"];

//                if(parameter_default.type() != parameter_value.type()){
//                    ImGui::Text("Error: different type Value/Default %s", parameter_c_str);
//                    continue;
//                }

                if (parameter_value.is_boolean()) {
                    has_changed |= ImGui::Checkbox(parameter.key().c_str(), parameter_value.get_ptr<bool *>());
                } else if (parameter_value.is_number()) {
                    auto &parameter_min = parameter.value()["min_value"];
                    auto &parameter_max = parameter.value()["max_value"];

                    if (parameter_value.is_number_float()) {
                        double min{0};
                        double max{1};
                        double value;
                        parameter_value.get_to(value);
                        parameter_min.get_to(min);
                        parameter_max.get_to(max);
                        if (ImGui::SliderScalar(parameter_c_str, ImGuiDataType_Double,
                                                &value, &min, &max)) {
                            parameter_value = value;
                            has_changed = true;
                        }
                    } else if (parameter_value.is_number_unsigned()) {
                        uint64_t min{0};
                        uint64_t max{1000};
                        uint64_t value;
                        parameter_value.get_to(value);
                        parameter_min.get_to(min);
                        parameter_max.get_to(max);
                        if (ImGui::SliderScalar(parameter_c_str, ImGuiDataType_U64,
                                                &value, &min, &max)) {
                            parameter_value = value;
                            has_changed = true;
                        }
                    } else {
                        int64_t min{0};
                        int64_t max{1000};
                        int64_t value;
                        parameter_value.get_to(value);
                        parameter_min.get_to(min);
                        parameter_max.get_to(max);
                        if (ImGui::SliderScalar(parameter_c_str, ImGuiDataType_S64,
                                                &value, &min, &max)) {
                            parameter_value = value;
                            has_changed = true;
                        }
                    }

                } else if (parameter_value.is_string()) {
                    auto has_enum_values = parameter.value().find("enum_values");
                    std::string &string_value = parameter_value.get_ref<std::string &>();
                    if (has_enum_values == parameter.value().end()) {
                        has_changed |=
                            ImGui::InputText(parameter.key().c_str(), parameter_value.get_ptr<std::string *>());
                    } else {
                        auto &enum_values = parameter.value()["enum_values"];
                        int selected{-1};
                        int prev_selected{-1};
                        std::vector<const char *> items;
                        items.reserve(enum_values.size());
                        for (int i = 0; i < enum_values.size(); ++i) {
                            auto &current_enum_value = enum_values.at(i).get_ref<std::string &>();
                            if (current_enum_value == string_value) {
                                selected = i;
                                prev_selected = i;
                            }
                            items.emplace_back(current_enum_value.c_str());
                        }
                        has_changed |= ImGui::Combo(parameter_c_str, &selected, items.data(), items.size());
                        if (prev_selected != selected && selected >= 0 && selected < items.size()) {
                            parameter_value = std::string(items[selected]);
                        }
                    }

                } else {
                    ImGui::Text("unknown parameter type");
                }

            }
        }

    }

    if (has_changed && onChange) {
        onChange({pattern_instance});
    }
}
} // traact