/** Copyright (C) 2022  Frieder Pankratz <frieder.pankratz@gmail.com> **/

#include "EditorElements.h"

namespace traact::gui::editor {

    EditorPattern::EditorPattern(pattern::instance::PatternInstance::Ptr patternInstance) : Pattern(patternInstance){

        DfgNode = std::make_shared<DFGNode>(Pattern);
        for (const auto& port_group : Pattern->port_groups) {
            for (const auto& group_instance : port_group) {


                if(group_instance->port_group.coordinate_systems.empty()){
                    continue;
                }


                const auto& nodes = group_instance->port_group.coordinate_systems;
                const auto& edges = group_instance->port_group.edges;




                auto create_srg_node = [this,&edges](const auto& name_node) {
                    auto is_output_node = [&edges](const pattern::spatial::CoordinateSystem& coordinate_system){
                        auto result = std::find_if(edges.cbegin(), edges.cend(), [&coordinate_system](auto& edge){
                            return coordinate_system.name == std::get<2>(edge);
                        });
                        return result != edges.cend();
                    };

                    bool is_output = is_output_node(name_node.second);
                    SrgNodes.emplace_back(new SRGNode(&name_node.second, is_output,
                                                      Pattern));
                };

                std::for_each(nodes.cbegin(), nodes.cend(), create_srg_node);

                auto create_srg_edge = [this](const auto& source_target_pin) {
                    auto find_coordinate_system = [this](const std::string& name) -> SRGNode::Ptr {
                        auto result = std::find_if(SrgNodes.begin(), SrgNodes.end(), [&name](auto& srg_node){
                            return srg_node->CoordinateSystem->name == name;
                        });
                        if(result != SrgNodes.end())
                            return *result;
                        return nullptr;
                    };

                    auto find_port = [this](const std::string& name) -> DFGPin::Ptr {
                        auto if_pin_name = [&name](DFGPin::Ptr& pin){
                            return pin->TraactPort->getName() == name;
                        };
                        auto output_result = std::find_if(DfgNode->Outputs.begin(), DfgNode->Outputs.end(), if_pin_name);
                        if(output_result != DfgNode->Outputs.end())
                            return *output_result;
                        auto input_result = std::find_if(DfgNode->Inputs.begin(), DfgNode->Inputs.end(), if_pin_name);
                        if(input_result != DfgNode->Inputs.end())
                            return *input_result;
                        return nullptr;
                    };

                    const auto& source_name = std::get<0>(source_target_pin);
                    const auto& target_name = std::get<1>(source_target_pin);
                    const auto& port_name = std::get<2>(source_target_pin);



                    SRGNode::Ptr sourceNode = find_coordinate_system(source_name);
                    SRGNode::Ptr targetNode = find_coordinate_system(target_name);
                    DFGPin::Ptr dfgPin = find_port(port_name);

                    if(!sourceNode || !targetNode || !dfgPin ){
                        spdlog::error("could not find node elements for source/target node or dfgPin");
                        return;
                    }


                    bool isOutput = dfgPin->TraactPort->getPortType() == traact::pattern::PortType::PRODUCER;
                    auto newEdge = std::make_shared<SRGEdge>(sourceNode, targetNode, dfgPin, isOutput);
                    sourceNode->Edges.push_back(newEdge);
                    targetNode->Edges.push_back(newEdge);
                    SrgConnections.push_back(newEdge);
                };

                std::for_each(edges.cbegin(), edges.cend(), create_srg_edge);
            }
        }


    }
}