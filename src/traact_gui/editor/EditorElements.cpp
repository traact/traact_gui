/*  BSD 3-Clause License
 *
 *  Copyright (c) 2020, FriederPankratz <frieder.pankratz@gmail.com>
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions are met:
 *
 *  1. Redistributions of source code must retain the above copyright notice, this
 *     list of conditions and the following disclaimer.
 *
 *  2. Redistributions in binary form must reproduce the above copyright notice,
 *     this list of conditions and the following disclaimer in the documentation
 *     and/or other materials provided with the distribution.
 *
 *  3. Neither the name of the copyright holder nor the names of its
 *     contributors may be used to endorse or promote products derived from
 *     this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 *  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 *  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 *  DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 *  FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 *  DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 *  SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 *  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 *  OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 *  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
**/

#include "EditorElements.h"

namespace traact::gui::editor {

    EditorPattern::EditorPattern(pattern::instance::PatternInstance::Ptr patternInstance) : Pattern(patternInstance){

        DfgNode = std::make_shared<DFGNode>(Pattern);

        if(!(!Pattern->local_pattern.coordinate_systems_.empty() || ! Pattern->local_pattern.group_ports.empty()) )
            return;

        const auto& nodes = Pattern->local_pattern.coordinate_systems_;
        const auto& edges = Pattern->local_pattern.edges_;




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


            bool isOutput = dfgPin->TraactPort->GetPortType() == traact::pattern::PortType::Producer;
            auto newEdge = std::make_shared<SRGEdge>(sourceNode, targetNode, dfgPin, isOutput);
            sourceNode->Edges.push_back(newEdge);
            targetNode->Edges.push_back(newEdge);
            SrgConnections.push_back(newEdge);
        };

        std::for_each(edges.cbegin(), edges.cend(), create_srg_edge);
    }
}