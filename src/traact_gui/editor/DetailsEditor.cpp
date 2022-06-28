/** Copyright (C) 2022  Frieder Pankratz <frieder.pankratz@gmail.com> **/

#include "DetailsEditor.h"
#include "external/imgui_misc/imgui_stdlib.h"
namespace traact {
void gui::DetailsEditor::operator()(std::shared_ptr<DataflowFile> &dataflow_file) const {
    if(!dataflow_file){
        return;
    }
    auto& graph_instance = dataflow_file->graph_editor_.Graph;
    ImGui::InputText("Name", &graph_instance->name);

    for (auto& time_domain : graph_instance->timedomain_configs) {
        ImGui::Text("Time Domain %lu",time_domain.first);
        auto& time_domain_config = time_domain.second;
        double min_freq = 0;
        double max_freq = 100;
        ImGui::SliderScalar("Sensor frequency",ImGuiDataType_Double,&time_domain_config.sensor_frequency, &min_freq, &max_freq);
        ImGui::DragInt("Worker Count",&time_domain_config.cpu_count, 0.1f, -std::thread::hardware_concurrency(), 100);
    }


}
void gui::DetailsEditor::operator()(std::shared_ptr<traact::pattern::instance::PatternInstance> &pattern_instance) const {
    if(!pattern_instance){
        return;
    }
    bool has_changed{false};
    ImGui::Text("%s", pattern_instance->instance_id.c_str());
    for (auto& [group_name, group_index] : pattern_instance->port_group_name_to_index) {
        ImGui::Text("%s", group_name.c_str());
        for (auto& group_instance : pattern_instance->port_groups[group_index]) {
            ImGui::Text("Group instance %d", group_instance->port_group_instance_index);
            for (auto& parameter: group_instance->port_group.parameter.items()) {
                const auto* parameter_c_str = parameter.key().c_str();
                auto& parameter_value = parameter.value()["value"];
                auto& parameter_default = parameter.value()["default"];

                if(parameter_default.type() != parameter_value.type()){
                    ImGui::Text("Error: different type Value/Default %s", parameter_c_str);
                    continue;
                }

                if(parameter_value.is_boolean()){
                    has_changed |= ImGui::Checkbox(parameter.key().c_str(), parameter_value.get_ptr<bool*>());
                } else if(parameter_value.is_number()) {
                    auto& parameter_min = parameter.value()["min_value"];
                    auto& parameter_max = parameter.value()["max_value"];
                    if(parameter_default.type() != parameter_min.type() || parameter_default.type() != parameter_max.type()){
                        ImGui::Text("Error: different type Default/Min/Max %s", parameter_c_str);
                        continue;
                    }

                    if(parameter_value.is_number_float()){
                        has_changed |=ImGui::SliderScalar(parameter_c_str, ImGuiDataType_Double,parameter_value.get_ptr<double*>(), parameter_min.get_ptr<double*>(), parameter_max.get_ptr<double*>() );
                    } else if(parameter_value.is_number_unsigned()){
                        has_changed |=ImGui::SliderScalar(parameter_c_str, ImGuiDataType_U64,parameter_value.get_ptr<uint64_t *>(), parameter_min.get_ptr<uint64_t*>(), parameter_max.get_ptr<uint64_t*>() );
                    } else {
                        has_changed |=ImGui::SliderScalar(parameter_c_str, ImGuiDataType_S64,parameter_value.get_ptr<int64_t *>(), parameter_min.get_ptr<int64_t*>(), parameter_max.get_ptr<int64_t*>() );
                    }
                } else if(parameter_value.is_string()){
                    auto has_enum_values = parameter.value().find("enum_values");
                    std::string& string_value = parameter_value.get_ref<std::string&>();
                    if(has_enum_values == parameter.value().end()){
                        has_changed |= ImGui::InputText(parameter.key().c_str(), parameter_value.get_ptr<std::string*>());
                    } else {
                        auto& enum_values = parameter.value()["enum_values"];
                        int selected{-1};
                        int prev_selected{-1};
                        std::vector<const char*> items;
                        items.reserve(enum_values.size());
                        for (int i = 0; i < enum_values.size(); ++i) {
                            auto& current_enum_value = enum_values.at(i).get_ref<std::string&>();
                            if(current_enum_value == string_value){
                                selected = i;
                                prev_selected = i;
                            }
                            items.emplace_back(current_enum_value.c_str());
                        }
                        has_changed |=ImGui::Combo(parameter_c_str, &selected, items.data(), items.size());
                        if(prev_selected != selected && selected >= 0 && selected < items.size()){
                            parameter_value = std::string(items[selected]);
                        }
                    }

                } else {
                    ImGui::Text("unknown parameter type");
                }

            }
        }

    }

    if(has_changed && onChange){
        onChange();
    }
}
} // traact