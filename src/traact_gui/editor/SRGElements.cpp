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

#include "SRGElements.h"
#include "EditorUtils.h"

namespace traact::gui::editor {
    SRGNode::SRGNode(const pattern::spatial::CoordinateSystem *coordinateSystem, bool isOutput,
                     const pattern::instance::PatternInstance::Ptr pattern) : ID(utils::GetNextId()),
                                            OutputPinID(utils::GetNextId()),
                                            InputPinID(utils::GetNextId()),
                                            CoordinateSystem(
                                                    coordinateSystem),
                                            IsOutput(
                                                    isOutput),
                                                    Pattern(pattern)
    {

    }



    ax::NodeEditor::PinId SRGNode::GetSourceID() const {
        if(Parent)
            return Parent->OutputPinID;

        return OutputPinID;
    }

    ax::NodeEditor::PinId SRGNode::GetTargetId() const {
        if(Parent)
            return Parent->InputPinID;
        return InputPinID;
    }

    bool SRGNode::IsVisible() {
        return Parent == nullptr;
    }

    std::string SRGNode::GetName() {

        return fmt::format("{0}:{1}",Pattern->instance_id,CoordinateSystem->name);
    }

    bool SRGNode::ConnectedTo(std::shared_ptr<SRGEdge> edge) {

        for(const auto& connected_port : edge->DfgPin->TraactPort->connectedToPtr()) {
            const auto connected_id = connected_port->getId();
            for(const auto& local_edges : Edges) {
                if(local_edges->DfgPin->TraactPort->getId() == connected_id)
                    return true;
            }
        }

        return false;
    }

    void SRGNode::SavePosition() {
        if(Parent)
            Position = ax::NodeEditor::GetNodePosition(Parent->ID);
        else
            Position = ax::NodeEditor::GetNodePosition(ID);

    }

    void SRGNode::RestorePosition() {
        ax::NodeEditor::SetNodePosition(ID, Position);
    }


    SRGEdge::SRGEdge(SRGNode::Ptr sourceNode, SRGNode::Ptr targetNode,
                     DFGPin::Ptr dfgPin, const bool isOutput) : ID(utils::GetNextId()), SourceNode(sourceNode), TargetNode(targetNode),
                                                             DfgPin(dfgPin), IsOutput(isOutput){
        //SourceNode->Edges.emplace_back(shared_from_this());
        //TargetNode->Edges.emplace_back(shared_from_this());

    }



    ax::NodeEditor::PinId SRGEdge::GetSourceID() const {
        return SourceNode->GetSourceID();
    }

    ax::NodeEditor::PinId SRGEdge::GetTargetId() const {
        return TargetNode->GetTargetId();
    }

    bool SRGEdge::IsVisible() const {
        return Parent == nullptr;
    }

    bool SRGEdge::isConnected() const{
        return DfgPin->TraactPort->isConnected();
    }


    void SRGMergedNode::Merge(SRGMergedNode::Ptr node) {
        std::for_each(node->Children.begin(), node->Children.end(), [this](const auto& n) {
            Merge(n);
        });
        node->Children.clear();

    }

    void SRGMergedNode::Merge(SRGNode::Ptr node) {
        node->Parent = shared_from_this();
        Children.push_back(node);
    }

    SRGMergedNode::SRGMergedNode() : Name(""), ID(utils::GetNextId()), OutputPinID(utils::GetNextId()), InputPinID(utils::GetNextId()){}

    void SRGMergedNode::SavePosition() {
        Position = ax::NodeEditor::GetNodePosition(ID);

    }

    void SRGMergedNode::RestorePosition() {
        ax::NodeEditor::SetNodePosition(ID, Position);
    }

    SRGMergedEdge::SRGMergedEdge() : ID(utils::GetNextId()), OutputPinID(utils::GetNextId()), InputPinID(utils::GetNextId()){}
}