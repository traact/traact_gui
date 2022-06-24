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

#ifndef TRAACTMULTI_SRGELEMENTS_H
#define TRAACTMULTI_SRGELEMENTS_H

#include "DFGElements.h"
#include <traact/pattern/instance/PatternInstance.h>
namespace traact::gui::editor {

    struct SRGEdge;
    struct SRGMergedEdge;
    struct SRGMergedNode;

    struct SRGNode : public std::enable_shared_from_this<SRGNode> {
        typedef std::shared_ptr<SRGNode> Ptr;
        ax::NodeEditor::NodeId ID;
        ax::NodeEditor::PinId OutputPinID;
        ax::NodeEditor::PinId InputPinID;
        const pattern::spatial::CoordinateSystem* CoordinateSystem;
        const pattern::instance::PatternInstance::Ptr Pattern;
        const bool IsOutput;
        bool IsVisible();
        std::vector<SRGNode::Ptr> MergedNodes;
        std::shared_ptr<SRGMergedNode> Parent{nullptr};
        std::vector<std::shared_ptr<SRGEdge>> Edges;

        ImVec2 Position;

        SRGNode(const pattern::spatial::CoordinateSystem *coordinateSystem, bool isOutput,
                const pattern::instance::PatternInstance::Ptr pattern);

        ax::NodeEditor::PinId GetSourceID() const;
        ax::NodeEditor::PinId GetTargetID() const;
        std::string GetName();
        bool ConnectedTo(std::shared_ptr<SRGEdge> edge);
        void SavePosition();
        void RestorePosition();

        //bool HasOtherConnections(SRGEdge *pEdge);

    };

    struct SRGEdge : public std::enable_shared_from_this<SRGEdge> {
        typedef std::shared_ptr<SRGEdge> Ptr;
        ax::NodeEditor::LinkId ID;
        SRGNode::Ptr SourceNode;
        SRGNode::Ptr TargetNode;
        DFGPin::Ptr DfgPin;
        const bool IsOutput;
        std::shared_ptr<SRGMergedEdge> Parent{nullptr};

        SRGEdge(SRGNode::Ptr sourceNode, SRGNode::Ptr targetNode,
                DFGPin::Ptr dfgPin, bool isOutput);

        bool IsVisible() const;


        ax::NodeEditor::PinId GetSourceID() const;
        ax::NodeEditor::PinId GetTargetID() const;


        bool IsConnected() const;


    };

    struct SRGMergedNode : public std::enable_shared_from_this<SRGMergedNode>{
        typedef typename std::shared_ptr<SRGMergedNode> Ptr;

        SRGMergedNode();

        std::string Name;
        ax::NodeEditor::NodeId ID;
        ax::NodeEditor::PinId OutputPinID;
        ax::NodeEditor::PinId InputPinID;

        std::vector<SRGNode::Ptr> Children;

        ImVec2 Position;

        void Merge(SRGMergedNode::Ptr node);
        void Merge(SRGNode::Ptr node);

        void SavePosition();
        void RestorePosition();

    };

    struct SRGMergedEdge : public std::enable_shared_from_this<SRGEdge>{
        typedef typename std::shared_ptr<SRGMergedEdge> Ptr;

        SRGMergedEdge();

        ax::NodeEditor::LinkId ID;
        ax::NodeEditor::PinId OutputPinID;
        ax::NodeEditor::PinId InputPinID;

        std::vector<SRGEdge::Ptr> Children;
    };
}

#endif //TRAACTMULTI_SRGELEMENTS_H
