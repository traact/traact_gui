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

#include "PatternGraphEditor.h"
#include "EditorUtils.h"
#include <traact_gui/ImGuiUtils.h>
namespace traact::gui::editor {

void PatternGraphEditor::CreateNodes() {
    Patterns.clear();
    DfgConnections.clear();
    SrgMergedEdges.clear();
    SrgMergedNodes.clear();
    for (auto& pattern : Graph->getAll()) {
        Patterns.emplace_back(std::make_shared<EditorPattern>(pattern));
    }
}

void PatternGraphEditor::CreateConnections() {
    DfgConnections.clear();
    for (const auto& pattern : Patterns) {
        const auto& node = pattern->DfgNode;
        for (const auto& input_pin : node->Inputs) {

            if(input_pin->TraactPort->IsConnected()){
                const auto output_pin = FindDFGOutputPin(input_pin->TraactPort->connected_to);
                if(!output_pin){
                    spdlog::error("node editor pin not found for {0}:{1}",input_pin->TraactPort->connected_to.first,input_pin->TraactPort->connected_to.second);
                    continue;
                }

                DfgConnections.emplace_back(std::make_shared<DFGLink>(output_pin->ID, input_pin->ID));
            }
        }
    }
    UpdateSRGGraph();
}

const DFGPin::Ptr PatternGraphEditor::FindDFGOutputPin(const pattern::instance::ComponentID_PortName &port) {
    for (const auto& pattern : Patterns) {
        const auto& node = pattern->DfgNode;
        for (const auto& output_pin : node->Outputs) {
            if(output_pin->TraactPort->getID() == port)
                return output_pin;
        }

    }
    return nullptr;
}

    DFGPin::Ptr PatternGraphEditor::FindPin(ax::NodeEditor::PinId id) {
    if (!id)
        return nullptr;

    for (auto& pattern : Patterns)
    {
        const auto& node = pattern->DfgNode;
        for (auto& pin : node->Inputs)
            if (pin->ID == id)
                return pin;

        for (auto& pin : node->Outputs)
            if (pin->ID == id)
                return pin;
    }

    return nullptr;
}

    DFGNode::Ptr PatternGraphEditor::FindNode(ax::NodeEditor::NodeId id)
{

    auto result = std::find_if(Patterns.begin(), Patterns.end(), [id](auto& pattern) { return pattern->DfgNode->ID == id; });
    if (result != Patterns.end())
        return (*result)->DfgNode;

    return nullptr;
}

    DFGLink::Ptr PatternGraphEditor::FindLink(ax::NodeEditor::LinkId id)
{
    auto result = std::find_if(DfgConnections.begin(), DfgConnections.end(), [id](auto& link) { return link->ID == id; });
    if (result != DfgConnections.end())
        return *result;

    return nullptr;
}
bool PatternGraphEditor::IsPinLinked(ax::NodeEditor::PinId id)
{
    if (!id)
        return false;

    auto result = std::find_if(DfgConnections.begin(), DfgConnections.end(), [id](auto& link) { return link->StartPinID == id || link->EndPinID == id; });
    if (result != DfgConnections.end())
        return true;

    return false;
}

std::optional<std::string> PatternGraphEditor::CanCreateLink(DFGPin::Ptr startPin, DFGPin::Ptr endPin)
{



    return Graph->checkSourceAndSinkConnectionError(startPin->TraactPort->getID(), endPin->TraactPort->getID());
}

void PatternGraphEditor::ConnectPins(ax::NodeEditor::PinId startPin, ax::NodeEditor::PinId endPin) {
    spdlog::info("connect {0} {1}", startPin.Get(), endPin.Get());


    if(IsPinLinked(endPin)){
        DisconnectPin(endPin);
    }

    auto source_pin = FindPin(startPin);
    auto sink_pin = FindPin(endPin);

    if(!source_pin || !sink_pin){
        spdlog::error("source or sink pin not found");
        return;
    }
    auto source_id = source_pin->TraactPort->getID();
    auto sink_id = sink_pin->TraactPort->getID();
    Graph->connect(source_id.first, source_id.second, sink_id.first, sink_id.second);

    DfgConnections.emplace_back(std::make_shared<DFGLink>(startPin, endPin));
    UpdateSRGGraph();
}

void PatternGraphEditor::DisconnectPin(ax::NodeEditor::LinkId id) {
    spdlog::info("disconnect {0}", id.Get());
    auto link = FindLink(id);
    if(link){
        DisconnectPin(link->EndPinID);
    }

    else
        spdlog::error("link to disconnect not found");
}

void PatternGraphEditor::DisconnectPin(ax::NodeEditor::PinId endPin) {
    auto result = std::find_if(DfgConnections.begin(), DfgConnections.end(), [endPin](auto& link) { return link->EndPinID == endPin; });
    if (result != DfgConnections.end()){

        auto source_pin = FindPin((*result)->StartPinID);
        auto sink_pin = FindPin((*result)->EndPinID);

        if(!sink_pin){
            spdlog::error("sink pin not found");
            return;
        }
        if(!sink_pin->TraactPort->IsConnected()) {
            spdlog::error("traact port is not connected");
            return;
        }
        auto sink_id = sink_pin->TraactPort->getID();
        Graph->disconnect(sink_id.first, sink_id.second);
        DfgConnections.erase(result);
        UpdateSRGGraph();
    }
    else
        spdlog::error("link to disconnect not found");
}

void PatternGraphEditor::DeleteNode(ax::NodeEditor::NodeId id) {
//                        auto id = std::find_if(m_Nodes.begin(), m_Nodes.end(), [nodeId](auto& node) { return node.ID == nodeId; });
//                        if (id != m_Nodes.end())
//                            m_Nodes.erase(id);
}

EditorPattern::Ptr PatternGraphEditor::CreatePatternInstance(pattern::Pattern::Ptr pattern) {
    auto allPattern = Graph->getAll();
    std::string pattern_id = "";
    for(int i=0;i<std::numeric_limits<int>::max();++i){
        pattern_id = fmt::format("pattern_{0}", i);
        auto result = std::find_if(allPattern.cbegin(),allPattern.cend(), [&pattern_id](const auto& value){
            return value->instance_id == pattern_id;
        });
        if(result == allPattern.cend())
            break;
    }
    auto pattern_instance = Graph->addPattern(pattern_id, pattern);

    Patterns.emplace_back(std::make_shared<EditorPattern>(pattern_instance));


    return Patterns.back();
}


void PatternGraphEditor::UpdateSRGGraph() {

   /* auto find_srg_edge_by_pin = [this](const ax::NodeEditor::PinId& pin_id)  {
        auto result = std::find_if(SrgMergedEdges.cbegin(),SrgMergedEdges.cend(), [&pin_id](const SRGEdge::Ptr& edge){
            return edge->DfgPin->ID == pin_id;
        });
        if(result != SrgMergedEdges.cend())
            return result;
        return SrgMergedEdges.cend();
    };

    auto update_srg_connection = [find_srg_edge_by_pin](const DFGLink::Ptr& dfg_link){
        auto output_edge = *find_srg_edge_by_pin(dfg_link->StartPinID);
        auto input_edge = *find_srg_edge_by_pin(dfg_link->EndPinID);
        if(!output_edge || !input_edge){
            spdlog::error("srg edge not found for dfg port");
            return;
        }
        output_edge->MergeEdge(input_edge);
    };

    auto disconnect_unconnected_edges = [find_srg_edge_by_pin, this](const SRGEdge::Ptr& edge){

        if(edge->IsConnected())
            return;

        if(edge->SourceNode->HasOtherConnections(edge))
            return;
        if(edge->TargetNode->HasOtherConnections(edge))
            return;

        edge->UntangleEdgeAndNodes();


    };

    std::vector<SRGEdge::Ptr> srg_connection_copy = SRGConnections;

    std::for_each(DFGConnections.cbegin(), DFGConnections.cend(), update_srg_connection);
    std::for_each(srg_connection_copy.begin(), srg_connection_copy.end(), disconnect_unconnected_edges);*/
}

    std::optional<std::string>
    PatternGraphEditor::CanMergeNodes(ax::NodeEditor::NodeId node1, ax::NodeEditor::NodeId node2) {
        auto result_merge_node1 = FindSrgMergeNode(node1);
        auto result_merge_node2 = FindSrgMergeNode(node2);

        std::vector<const pattern::spatial::CoordinateSystem*> source_cs;
        std::vector<const pattern::spatial::CoordinateSystem*> target_cs;
        source_cs.reserve(100);
        target_cs.reserve(100);

        auto add_all = [](std::vector<const pattern::spatial::CoordinateSystem*>& container, const SRGMergedNode::Ptr& mergedNode){
            for(const auto& node : mergedNode->Children)
                container.push_back(node->CoordinateSystem);
        };

        if(result_merge_node1){
            assert(result_merge_node1);
            add_all(source_cs, result_merge_node1);

        } else {
            auto result_pattern_node1 = FindSrgNode(node1);
            assert(result_pattern_node1);
            source_cs.push_back(result_pattern_node1->CoordinateSystem);
        }

        if(result_merge_node2) { // if both are merge nodes
            assert(result_merge_node2);
            add_all(target_cs, result_merge_node2);
        } else {
            auto result_pattern_node2 = FindSrgNode(node2);
            assert(result_pattern_node2);
            target_cs.push_back(result_pattern_node2->CoordinateSystem);
        }



        auto contains_any_cs = [this](const pattern::Pattern& pat, std::vector<const pattern::spatial::CoordinateSystem*>& cs) {
            for(const auto& local_cs : cs ) {
                for(const auto& name_cs : pat.coordinate_systems_) {
                    if(&name_cs.second == local_cs)
                        return true;
                }
            }
            return false;
        };

        for(const auto& pat : Patterns) {
            auto contains_source = contains_any_cs(pat->Pattern->local_pattern, source_cs);
            auto contains_target = contains_any_cs(pat->Pattern->local_pattern, target_cs);
            if(contains_source && contains_target){
                return {fmt::format("Pattern {0}: can not merge nodes of the same pattern", pat->Pattern->instance_id)};
            }
        }
        return std::optional<std::string>();
    }

    SRGMergedNode::Ptr PatternGraphEditor::MergeNodes(ax::NodeEditor::NodeId node1, ax::NodeEditor::NodeId node2) {


        auto result_merge_node1 = FindSrgMergeNode(node1);
        auto result_merge_node2 = FindSrgMergeNode(node2);

        if(result_merge_node1){
            if(result_merge_node2) { // if both are merge nodes, merge node 2 into node 1
                assert(result_merge_node1);
                assert(result_merge_node2);
                result_merge_node1->Merge((result_merge_node2));
                auto result_erase = std::find(SrgMergedNodes.begin(), SrgMergedNodes.end(), result_merge_node2);
                if(result_erase != SrgMergedNodes.end())
                    SrgMergedNodes.erase(result_erase);
            } else { // otherwise merge srg node 2 into merge node 1
                auto result_pattern_node2 = FindSrgNode(node2);
                assert(result_merge_node1);
                assert(result_pattern_node2);
                result_merge_node1->Merge(result_pattern_node2);

            }
        } else {
            auto result_pattern_node1 = FindSrgNode(node1);
            if(result_merge_node2) { // merge srg node 1 into merge node 2
                assert(result_pattern_node1);
                assert(result_merge_node2);
                result_merge_node2->Merge(result_pattern_node1);
            } else { // otherwise both are srg nodes, create new merge node from node 1
                auto result_pattern_node2 = FindSrgNode(node2);
                assert(result_pattern_node1);
                assert(result_pattern_node2);

                auto newMergeNode = std::make_shared<SRGMergedNode>();
                newMergeNode->Name = result_pattern_node1->CoordinateSystem->name;
                newMergeNode->Merge(result_pattern_node1);
                newMergeNode->Merge(result_pattern_node2);
                newMergeNode->Position = result_pattern_node2->Position;
                SrgMergedNodes.push_back(newMergeNode);
                return newMergeNode;

            }

        }

        return nullptr;

    }

    SRGMergedNode::Ptr PatternGraphEditor::FindSrgMergeNode(ax::NodeEditor::NodeId id) {
        auto result = std::find_if(SrgMergedNodes.cbegin(),SrgMergedNodes.cend(), [&id](const auto& value){
            return value->ID == id;
        });
        if(result != SrgMergedNodes.cend())
            return *result;
        return nullptr;
    }

    SRGNode::Ptr PatternGraphEditor::FindSrgNode(ax::NodeEditor::NodeId id) {
        for(const auto& pattern : Patterns){
            auto result = std::find_if(pattern->SrgNodes.cbegin(),pattern->SrgNodes.cend(), [&id](const auto& value){
                return value->ID == id;
            });
            if(result != pattern->SrgNodes.cend())
                return *result;
        }

        return nullptr;
    }

    void PatternGraphEditor::SplitNode(ax::NodeEditor::NodeId id) {
        auto merge_node = FindSrgMergeNode(id);
        assert(merge_node);

        // split merge node into unrelated nodes (e.g. original srg nodes or new merge nodes)
        // all nodes in the child list are from different patterns
        // two nodes are unrelated when they do not share a connected edge

        std::vector<std::set<SRGNode::Ptr> > new_node_groups;

        auto add_or_create_node_group = [&new_node_groups](const SRGNode::Ptr& node){
            bool free_node = true;
            for(auto& edge : node->Edges) {
                if(!edge->IsConnected())
                    continue;
                free_node = false;

                auto result_groups = std::find_if(new_node_groups.begin(), new_node_groups.end(), [&edge, &node](const auto& node_group) {

                    for(const auto& local_node : node_group) {
                        if(node == local_node)
                            return true;
                        if(local_node->ConnectedTo(edge))
                            return true;
                    }
                   return false;
                });

                if(result_groups == new_node_groups.end()) {
                    new_node_groups.push_back({node});
                } else {
                    (*result_groups).emplace(node);
                }
            }
            if(free_node)
                new_node_groups.push_back({node});
        };

        for(const auto& node : merge_node->Children){
            add_or_create_node_group(node);
        }

        assert(new_node_groups.size() > 0);


        const float split_factor = M_PI / new_node_groups.size();
        const float split_distance = 60;



        for(std::size_t i=0;i<new_node_groups.size();++i){
            if(new_node_groups[i].size() > 1){
                auto newMergeNode = std::make_shared<SRGMergedNode>();

                for(const auto& tmp : new_node_groups[i]){
                    newMergeNode->Name = tmp->CoordinateSystem->name;
                    newMergeNode->Merge(tmp);
                }

                newMergeNode->Position = merge_node->Position + ImVec2(std::sin(split_factor*i)*split_distance, std::cos(split_factor*i)*split_distance);
                newMergeNode->RestorePosition();

                SrgMergedNodes.push_back(newMergeNode);
            } else {
                auto local_node = (*new_node_groups[i].begin());
                local_node->Parent = nullptr;

                local_node->Position = merge_node->Position + ImVec2(std::sin(split_factor*i)*split_distance, std::cos(split_factor*i)*split_distance);
                local_node->RestorePosition();
            }

        }

        auto result_merge_node = std::remove(SrgMergedNodes.begin(),SrgMergedNodes.end(), merge_node);
        SrgMergedNodes.erase(result_merge_node);





    }

}