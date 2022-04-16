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

#include "NodeEditorElements.h"
#include <spdlog/spdlog.h>

#include <utility>
#include <external/imgui-node-editor/imgui_node_editor.h>

namespace traact::gui {

    int GetNextId()
    {
        static int nextId = 0;
        return nextId++;
    }

    Pin::Pin(int id,  pattern::instance::PortInstance::Ptr port, Node* node) :
    ID(id),TraactPort(port), ParentNode(node){
    }

    Node::Node(int id, const pattern::instance::PatternInstance::Ptr& pattern) :
                ID(id), Pattern(pattern), Size(0, 0) {
        Inputs.reserve(Pattern->consumer_ports.size());
        for (auto& port : Pattern->consumer_ports) {
            Inputs.emplace_back(std::make_shared<Pin>(GetNextId(), &port, this));
        }

        max_output_name_length = 0;

        Outputs.reserve(Pattern->producer_ports.size());
        for (auto& port : Pattern->producer_ports) {
            max_output_name_length = std::max(max_output_name_length, port.getName().length());
            Outputs.emplace_back(std::make_shared<Pin>(GetNextId(), &port, this));
        }
        }

    void Node::UpdateOutputWeight() {
        OutputWeight = 0;
        for (const auto& output_pin : Outputs) {
            OutputWeight += output_pin->TraactPort->connectedToPtr().size();
        }

    }

    void Node::UpdateInputWeight() {
        InputWeight = 0;
        for (const auto& input : Inputs) {
            InputWeight += input->TraactPort->connectedToPtr().size();
        }
    }


    void PatternGraphEditor::CreateNodes() {
        DFGNodes.clear();
        SRGNodes.clear();
        SRGConnections.clear();
        for (auto& pattern : Graph->getAll()) {
            DFGNodes.emplace_back(std::make_shared<Node>(GetNextId(),pattern));
            CreateSRGPattern(DFGNodes.back());
        }
    }

    void PatternGraphEditor::CreateConnections() {
        DFGConnections.clear();
        for (const auto& node : DFGNodes) {
            for (const auto& input_pin : node->Inputs) {

                if(input_pin->TraactPort->IsConnected()){
                    const auto output_pin = FindDFGOutputPin(input_pin->TraactPort->connected_to);
                    if(!output_pin){
                        spdlog::error("node editor pin not found for {0}:{1}",input_pin->TraactPort->connected_to.first,input_pin->TraactPort->connected_to.second);
                        continue;
                    }
                    DFGConnections.emplace_back(std::make_shared<Link>(GetNextId(), output_pin->ID, input_pin->ID));
                }
            }
        }
        UpdateSRGGraph();
    }

    const Pin::Ptr PatternGraphEditor::FindDFGOutputPin(const pattern::instance::ComponentID_PortName &port) {
        for (const auto& node : DFGNodes) {
            for (const auto& output_pin : node->Outputs) {
                if(output_pin->TraactPort->getID() == port)
                    return output_pin;
            }

        }
        return nullptr;
    }

    Pin::Ptr PatternGraphEditor::FindPin(ax::NodeEditor::PinId id) {
        if (!id)
            return nullptr;

        for (auto& node : DFGNodes)
        {
            for (auto& pin : node->Inputs)
                if (pin->ID == id)
                    return pin;

            for (auto& pin : node->Outputs)
                if (pin->ID == id)
                    return pin;
        }

        return nullptr;
    }

    Node::Ptr PatternGraphEditor::FindNode(ax::NodeEditor::NodeId id)
    {

        auto result = std::find_if(DFGNodes.begin(), DFGNodes.end(), [id](auto& node) { return node->ID == id; });
        if (result != DFGNodes.end())
            return *result;

        return nullptr;
    }

    Link::Ptr PatternGraphEditor::FindLink(ax::NodeEditor::LinkId id)
    {
        auto result = std::find_if(DFGConnections.begin(), DFGConnections.end(), [id](auto& link) { return link->ID == id; });
        if (result != DFGConnections.end())
            return *result;

        return nullptr;
    }
    bool PatternGraphEditor::IsPinLinked(ax::NodeEditor::PinId id)
    {
        if (!id)
            return false;

        auto result = std::find_if(DFGConnections.begin(), DFGConnections.end(), [id](auto& link) { return link->StartPinID == id || link->EndPinID == id; });
        if (result != DFGConnections.end())
            return true;

        return false;
    }

    std::optional<std::string> PatternGraphEditor::CanCreateLink(Pin::Ptr startPin, Pin::Ptr endPin)
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

        DFGConnections.emplace_back(std::make_shared<Link>(GetNextId(), startPin, endPin));
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
        auto result = std::find_if(DFGConnections.begin(), DFGConnections.end(), [endPin](auto& link) { return link->EndPinID == endPin; });
        if (result != DFGConnections.end()){

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
            DFGConnections.erase(result);
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

    ax::NodeEditor::NodeId PatternGraphEditor::CreatePatternInstance(pattern::Pattern::Ptr pattern) {
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
        auto new_id = GetNextId();

        DFGNodes.emplace_back(std::make_shared<Node>(new_id,pattern_instance));
        CreateSRGPattern(DFGNodes.back());

        return new_id;
    }

    void PatternGraphEditor::CreateSRGPattern(Node::Ptr dfg_node) {

        if(!(!dfg_node->Pattern->local_pattern.coordinate_systems_.empty() || ! dfg_node->Pattern->local_pattern.group_ports.empty()) )
            return;

        const auto& nodes = dfg_node->Pattern->local_pattern.coordinate_systems_;
        const auto& edges = dfg_node->Pattern->local_pattern.edges_;

        std::vector<SRGNode::Ptr> new_SRGNodes;



        auto create_srg_node = [this,&new_SRGNodes,&edges](const auto& name_node) {
            auto is_output_node = [&edges](const pattern::spatial::CoordinateSystem& coordinate_system){
                auto result = std::find_if(edges.cbegin(), edges.cend(), [&coordinate_system](auto& edge){
                    return coordinate_system.name == std::get<2>(edge);
                });
                return result != edges.cend();
            };

            bool is_output = is_output_node(name_node.second);
            SRGNodes.template emplace_back(new SRGNode(GetNextId(), GetNextId(), GetNextId(), &name_node.second, is_output ));
            new_SRGNodes.push_back(SRGNodes.back());
        };

        std::for_each(nodes.cbegin(), nodes.cend(), create_srg_node);

        auto create_srg_edge = [this,&dfg_node, &new_SRGNodes](const auto& source_target_pin) {
            auto find_coordinate_system = [&new_SRGNodes](const std::string& name) -> SRGNode::Ptr {
                auto result = std::find_if(new_SRGNodes.begin(), new_SRGNodes.end(), [&name](auto& srg_node){
                    return srg_node->CoordinateSystem->name == name;
                });
                if(result != new_SRGNodes.end())
                    return *result;
                return nullptr;
            };

            auto find_port = [&dfg_node](const std::string& name) -> Pin::Ptr {
                auto if_pin_name = [&name](Pin::Ptr& pin){
                    return pin->TraactPort->getName() == name;
                };
                auto output_result = std::find_if(dfg_node->Outputs.begin(), dfg_node->Outputs.end(), if_pin_name);
                if(output_result != dfg_node->Outputs.end())
                    return *output_result;
                auto input_result = std::find_if(dfg_node->Inputs.begin(), dfg_node->Inputs.end(), if_pin_name);
                if(input_result != dfg_node->Inputs.end())
                    return *input_result;
                return nullptr;
            };

            const auto& source_name = std::get<0>(source_target_pin);
            const auto& target_name = std::get<1>(source_target_pin);
            const auto& port_name = std::get<2>(source_target_pin);



            SRGNode::Ptr sourceNode = find_coordinate_system(source_name);
            SRGNode::Ptr targetNode = find_coordinate_system(target_name);
            Pin::Ptr dfgPin = find_port(port_name);

            if(!sourceNode || !targetNode || !dfgPin ){
                spdlog::error("could not find node elements for source/target node or dfgPin");
                return;
            }

            bool isOutput = dfgPin->TraactPort->GetPortType() == traact::pattern::PortType::Producer;
            SRGConnections.template emplace_back(new SRGEdge(GetNextId(), sourceNode, targetNode, dfgPin, isOutput));
        };

        std::for_each(edges.cbegin(), edges.cend(), create_srg_edge);

    }

    void PatternGraphEditor::UpdateSRGGraph() {

        auto find_srg_edge_by_pin = [this](const ax::NodeEditor::PinId& pin_id)  {
            auto result = std::find_if(SRGConnections.cbegin(),SRGConnections.cend(), [&pin_id](const SRGEdge::Ptr& edge){
                return edge->DFGPin->ID == pin_id;
            });
            if(result != SRGConnections.cend())
                return result;
            return SRGConnections.cend();
        };

        auto update_srg_connection = [find_srg_edge_by_pin](const Link::Ptr& dfg_link){
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
        std::for_each(srg_connection_copy.begin(), srg_connection_copy.end(), disconnect_unconnected_edges);
    }

    SRGNode::SRGNode(const ax::NodeEditor::NodeId &id, const ax::NodeEditor::PinId &input_pinId,
                     const ax::NodeEditor::PinId &output_pinId,
                     const pattern::spatial::CoordinateSystem *coordinateSystem,
                     const bool isOutput) : ID(id),
                                                                                                        OutputPinID(output_pinId),
                                                                                                        InputPinID(input_pinId),
                                                                                                        CoordinateSystem(
                                                                                                                coordinateSystem),
                                                                                                        IsOutput(
                                                                                                                isOutput)
                                                                                                                {

    }

    void SRGNode::MergeNode(SRGNode::Ptr node) {
        if(node->Parent == this){
            return;
        } else if(node->Parent){
            MergeNode(node->Parent);
        } else {
            node->Parent = this;
            MergedNodes.emplace_back(node);
            if(!node->MergedNodes.empty()){
                auto& other_nodes = node->MergedNodes;
                std::for_each(other_nodes.begin(), other_nodes.end(), [this](auto& value){
                    value->Parent = this;
                });
                MergedNodes.insert(MergedNodes.cend(), other_nodes.begin(), other_nodes.end());
                other_nodes.clear();

            }


        }

    }

    ax::NodeEditor::PinId SRGNode::GetSourceID() const {
        if(Parent)
            return Parent->GetSourceID();

        return OutputPinID;
    }

    ax::NodeEditor::PinId SRGNode::GetTargetID() const {
        if(Parent)
            return Parent->GetTargetID();
        return InputPinID;
    }

    bool SRGNode::IsVisible() {
        return Parent == nullptr;
    }

    bool SRGNode::HasOtherConnections(SRGEdge *pEdge) {
        auto is_other_connected = [pEdge](const SRGEdge* edge){
            if(edge == pEdge)
                return false;
            return edge->IsConnected();
        };
        auto& source_edges = pEdge->SourceNode->Edges;
        auto source_result = std::find_if(source_edges.cbegin(), source_edges.cend(), is_other_connected);
        if(source_result != source_edges.cend())
            return true;

        auto& target_edges = pEdge->TargetNode->Edges;
        auto target_result = std::find_if(target_edges.cbegin(), target_edges.cend(), is_other_connected);
        if(target_result != target_edges.cend())
            return true;

        return false;
    }

    void SRGNode::UntangleNode() {
        spdlog::info("untangle node {0} {1}", ID.Get(), this->CoordinateSystem->name);
        if(Parent){
            auto result = std::find(Parent->MergedNodes.begin(), Parent->MergedNodes.end(), this);
            if(result != Parent->MergedNodes.end())
                Parent->MergedNodes.erase(result);

            Parent = nullptr;
        }

        if(!MergedNodes.empty()){
            SRGNode* front = MergedNodes.front();

            front->Parent = nullptr;
            std::for_each(MergedNodes.begin()+1, MergedNodes.end(), [&front](SRGNode* other){
                other->Parent = front;
                front->MergedNodes.emplace_back(other);
            });
            MergedNodes.clear();
        }
    }

    SRGEdge::SRGEdge(const ax::NodeEditor::LinkId &id, SRGNode::Ptr sourceNode, SRGNode::Ptr targetNode,
                     Pin::Ptr dfgPin, const bool isOutput) : ID(id), SourceNode(sourceNode), TargetNode(targetNode),
                                                        DFGPin(dfgPin), IsOutput(isOutput){
        SourceNode->Edges.emplace_back(this);
        TargetNode->Edges.emplace_back(this);

    }

    void SRGEdge::MergeEdge(SRGEdge::Ptr edge) {
        if(edge->Parent == this){
            return;
        } else if(edge->Parent){
            MergeEdge(edge->Parent);
        } else {
            edge->Parent = this;
            SourceNode->MergeNode(edge->SourceNode);
            TargetNode->MergeNode(edge->TargetNode);
            MergedEdges.emplace_back(edge);
            if(!edge->MergedEdges.empty()){
                auto& other_nodes = edge->MergedEdges;
                std::for_each(other_nodes.begin(), other_nodes.end(), [this](auto& value){
                    value->Parent = this;
                });
                MergedEdges.insert(MergedEdges.cend(), other_nodes.begin(), other_nodes.end());
                other_nodes.clear();

            }
        }



    }

    ax::NodeEditor::PinId SRGEdge::GetSourceID() const {
        return GetRootSourceNode()->GetSourceID();
    }

    ax::NodeEditor::PinId SRGEdge::GetTargetID() const {
        return GetRootTargetNode()->GetTargetID();
    }

    bool SRGEdge::IsVisible() const {
        return Parent == nullptr;
    }

    SRGNode::Ptr SRGEdge::GetRootSourceNode() const {
        if(SourceNode->Parent)
            return SourceNode->Parent;
        return SourceNode;
    }

    SRGNode::Ptr SRGEdge::GetRootTargetNode() const {
        if(TargetNode->Parent)
            return TargetNode->Parent;
        return TargetNode;
    }

    bool SRGEdge::IsConnected() const{
        return DFGPin->TraactPort->IsConnected();
    }

    void SRGEdge::UntangleEdgeAndNodes() {

        spdlog::info("untangle edge {0} {1}", ID.Get(), DFGPin->TraactPort->getName());
        if(Parent){
            auto result = std::find(Parent->MergedEdges.begin(), Parent->MergedEdges.end(), this);
            if(result != Parent->MergedEdges.end())
                Parent->MergedEdges.erase(result);

            Parent = nullptr;
        }

        if(!MergedEdges.empty()){
            SRGEdge* front = MergedEdges.front();

            front->Parent = nullptr;
            std::for_each(MergedEdges.begin()+1, MergedEdges.end(), [&front](SRGEdge* other){
                other->Parent = front;
                front->MergedEdges.emplace_back(other);
            });
            MergedEdges.clear();
        }

        SourceNode->UntangleNode();
        TargetNode->UntangleNode();

    }
};