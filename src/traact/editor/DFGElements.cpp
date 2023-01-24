/** Copyright (C) 2022  Frieder Pankratz <frieder.pankratz@gmail.com> **/

#include "DFGElements.h"
#include "EditorUtils.h"
namespace traact::gui::editor {

    DFGLink::DFGLink(ax::NodeEditor::PinId startPinId, ax::NodeEditor::PinId endPinId):
    ID(utils::GetNextId()), StartPinID(startPinId), EndPinID(endPinId), Color(255, 255, 255)
    {
    }

    DFGPin::DFGPin(const pattern::instance::PortInstance * port, DFGNode *node) :
            ID(utils::GetNextId()),TraactPort(port), ParentNode(node){
    }

    DFGNode::DFGNode(const pattern::instance::PatternInstance::Ptr& pattern) :
            ID(utils::GetNextId()), Pattern(pattern), Size(0, 0) {

        Inputs.reserve(Pattern->getConsumerPorts(0).size());
        for (const auto& port : Pattern->getConsumerPorts(0)) {
            Inputs.emplace_back(std::make_shared<DFGPin>( port, this));
        }

        max_output_name_length = 0;

        Outputs.reserve(Pattern->getProducerPorts(0).size());
        for (auto& port : Pattern->getProducerPorts(0)) {
            max_output_name_length = std::max(max_output_name_length, port->getName().length());
            Outputs.emplace_back(std::make_shared<DFGPin>(port, this));
        }
    }

    void DFGNode::UpdateOutputWeight() {
        OutputWeight = 0;
        for (const auto& output_pin : Outputs) {
            OutputWeight += output_pin->TraactPort->connectedToPtr().size();
        }

    }

    void DFGNode::UpdateInputWeight() {
        InputWeight = 0;
        for (const auto& input : Inputs) {
            InputWeight += input->TraactPort->connectedToPtr().size();
        }
    }

    void DFGNode::SavePosition() {
        Position = ax::NodeEditor::GetNodePosition(ID);
    }

    void DFGNode::RestorePosition() {
        ax::NodeEditor::SetNodePosition(ID, Position);
    }
const std::string &DFGNode::getName() const {
    return Pattern->instance_id;
}
}