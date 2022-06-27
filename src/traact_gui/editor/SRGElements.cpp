/** Copyright (C) 2022  Frieder Pankratz <frieder.pankratz@gmail.com> **/

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