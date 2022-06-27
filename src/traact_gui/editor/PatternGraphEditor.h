/** Copyright (C) 2022  Frieder Pankratz <frieder.pankratz@gmail.com> **/

#ifndef TRAACTMULTI_PATTERNGRAPHEDITOR_H
#define TRAACTMULTI_PATTERNGRAPHEDITOR_H

#include "SRGElements.h"
#include "DFGElements.h"
#include "EditorElements.h"
#include <traact/pattern/instance/GraphInstance.h>

namespace traact::gui::editor {

    struct PatternGraphEditor {
        pattern::instance::GraphInstance::Ptr Graph;
        std::vector<EditorPattern::Ptr> Patterns;

        std::vector<DFGLink::Ptr> DfgConnections;

        std::vector<SRGMergedNode::Ptr> SrgMergedNodes;
        std::vector<SRGMergedEdge::Ptr> SrgMergedEdges;

        void CreateNodes();
        void CreateConnections();
        const DFGPin::Ptr FindDFGOutputPin(const pattern::instance::ComponentID_PortName &port);

        DFGPin::Ptr FindPin(ax::NodeEditor::PinId id);
        DFGNode::Ptr FindNode(ax::NodeEditor::NodeId id);

        DFGLink::Ptr FindLink(ax::NodeEditor::LinkId id);
        bool IsPinLinked(ax::NodeEditor::PinId id);

        std::optional<std::string> CanCreateLink(DFGPin::Ptr startPin, DFGPin::Ptr endPin);

        void ConnectPins(ax::NodeEditor::PinId startPin, ax::NodeEditor::PinId endPin);
        void DisconnectPin(ax::NodeEditor::LinkId id);
        void DisconnectPin(ax::NodeEditor::PinId endPin);

        void DeleteNode(ax::NodeEditor::NodeId id);

        EditorPattern::Ptr  CreatePatternInstance(pattern::Pattern::Ptr pattern);

        void UpdateSRGGraph();

        std::optional<std::string> CanMergeNodes(ax::NodeEditor::NodeId node1, ax::NodeEditor::NodeId node2);
        SRGMergedNode::Ptr MergeNodes(ax::NodeEditor::NodeId node1, ax::NodeEditor::NodeId node2);

        SRGMergedNode::Ptr FindSrgMergeNode(ax::NodeEditor::NodeId id);
        SRGNode::Ptr FindSrgNode(ax::NodeEditor::NodeId id);
        void SplitNode(ax::NodeEditor::NodeId id);
    };
}


#endif //TRAACTMULTI_PATTERNGRAPHEDITOR_H
