/** Copyright (C) 2022  Frieder Pankratz <frieder.pankratz@gmail.com> **/

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
        ax::NodeEditor::PinId GetTargetId() const;
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
        ax::NodeEditor::PinId GetTargetId() const;


        bool isConnected() const;


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
