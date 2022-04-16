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

#ifndef TRAACTMULTI_NODEEDITORELEMENTS_H
#define TRAACTMULTI_NODEEDITORELEMENTS_H

#include <external/imgui-node-editor/imgui_node_editor.h>
#include <traact/pattern/instance/GraphInstance.h>
#include <traact/pattern/Pattern.h>
namespace traact::gui {

    int GetNextId();

    enum class NodeType {
        SRG,
        DFG
    };

    struct Link
    {
        typedef typename std::shared_ptr<Link> Ptr;
        ax::NodeEditor::LinkId ID;

        ax::NodeEditor::PinId StartPinID;
        ax::NodeEditor::PinId EndPinID;

        ImColor Color;

        Link(ax::NodeEditor::LinkId id, ax::NodeEditor::PinId startPinId, ax::NodeEditor::PinId endPinId):
                ID(id), StartPinID(startPinId), EndPinID(endPinId), Color(255, 255, 255)
        {
        }
    };



    struct Node;

    struct Pin {
        typedef typename std::shared_ptr<Pin> Ptr;
        ax::NodeEditor::PinId ID;
        Node *ParentNode;

        pattern::instance::PortInstance::Ptr TraactPort;

        Pin(int id,  pattern::instance::PortInstance::Ptr port, Node* node);
    };

    struct Node {
        typedef typename std::shared_ptr<Node> Ptr;
        ax::NodeEditor::NodeId ID;
        pattern::instance::PatternInstance::Ptr Pattern;

        std::vector <Pin::Ptr> Inputs;
        std::vector <Pin::Ptr> Outputs;

        ImVec2 Size;
        std::size_t max_output_name_length;

        std::string State;
        std::string SavedState;

        std::size_t OutputWeight;
        std::size_t InputWeight;

        Node(int id, const pattern::instance::PatternInstance::Ptr& pattern);
        void UpdateOutputWeight();
        void UpdateInputWeight();
    };

    struct SRGEdge;

    struct SRGNode {
        typedef SRGNode* Ptr;
        ax::NodeEditor::NodeId ID;
        ax::NodeEditor::PinId OutputPinID;
        ax::NodeEditor::PinId InputPinID;
        const pattern::spatial::CoordinateSystem* CoordinateSystem;
        const bool IsOutput;
        bool IsVisible();
        std::vector<SRGNode::Ptr> MergedNodes;
        SRGNode::Ptr Parent{nullptr};
        std::vector<SRGEdge*> Edges;

        SRGNode(const ax::NodeEditor::NodeId &id, const ax::NodeEditor::PinId &input_pinId,
                const ax::NodeEditor::PinId &output_pinId, const pattern::spatial::CoordinateSystem *coordinateSystem,
                bool isOutput);

        void MergeNode(SRGNode::Ptr node);
        ax::NodeEditor::PinId GetSourceID() const;
        ax::NodeEditor::PinId GetTargetID() const;

        bool HasOtherConnections(SRGEdge *pEdge);

        void UntangleNode();
    };

    struct SRGEdge {
        typedef SRGEdge* Ptr;
        ax::NodeEditor::LinkId ID;
        SRGNode::Ptr SourceNode;
        SRGNode::Ptr TargetNode;
        Pin::Ptr DFGPin;
        const bool IsOutput;
        SRGEdge::Ptr Parent{nullptr};

        SRGEdge(const ax::NodeEditor::LinkId &id, SRGNode::Ptr sourceNode, SRGNode::Ptr targetNode,
                Pin::Ptr dfgPin, bool isOutput);

        bool IsVisible() const;
        std::vector<SRGEdge::Ptr> MergedEdges;

        void MergeEdge(SRGEdge::Ptr edge);
        SRGNode::Ptr GetRootSourceNode() const;
        SRGNode::Ptr GetRootTargetNode() const;
        ax::NodeEditor::PinId GetSourceID() const;
        ax::NodeEditor::PinId GetTargetID() const;


        bool IsConnected() const;

        void UntangleEdgeAndNodes();
    };


    struct PatternGraphEditor {
        pattern::instance::GraphInstance::Ptr Graph;
        std::vector<Node::Ptr> DFGNodes;
        std::vector<Link::Ptr> DFGConnections;

        std::vector<SRGNode::Ptr> SRGNodes;
        std::vector<SRGEdge::Ptr> SRGConnections;

        void CreateNodes();
        void CreateConnections();
        const Pin::Ptr FindDFGOutputPin(const pattern::instance::ComponentID_PortName &port);

        Pin::Ptr FindPin(ax::NodeEditor::PinId id);
        Node::Ptr FindNode(ax::NodeEditor::NodeId id);

        Link::Ptr FindLink(ax::NodeEditor::LinkId id);
        bool IsPinLinked(ax::NodeEditor::PinId id);

        std::optional<std::string> CanCreateLink(Pin::Ptr startPin, Pin::Ptr endPin);

        void ConnectPins(ax::NodeEditor::PinId startPin, ax::NodeEditor::PinId endPin);

        void DisconnectPin(ax::NodeEditor::LinkId id);
        void DisconnectPin(ax::NodeEditor::PinId endPin);

        void DeleteNode(ax::NodeEditor::NodeId id);

        ax::NodeEditor::NodeId CreatePatternInstance(pattern::Pattern::Ptr pattern);

        void CreateSRGPattern(Node::Ptr dfg_node);
        void UpdateSRGGraph();
    };
}

#endif //TRAACTMULTI_NODEEDITORELEMENTS_H
