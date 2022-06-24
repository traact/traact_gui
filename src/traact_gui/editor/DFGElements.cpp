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

#include "DFGElements.h"
#include "EditorUtils.h"
namespace traact::gui::editor {

    DFGLink::DFGLink(ax::NodeEditor::PinId startPinId, ax::NodeEditor::PinId endPinId):
    ID(utils::GetNextId()), StartPinID(startPinId), EndPinID(endPinId), Color(255, 255, 255)
    {
    }

    DFGPin::DFGPin(pattern::instance::PortInstance::Ptr port, DFGNode* node) :
            ID(utils::GetNextId()),TraactPort(port), ParentNode(node){
    }

    DFGNode::DFGNode(const pattern::instance::PatternInstance::Ptr& pattern) :
            ID(utils::GetNextId()), Pattern(pattern), Size(0, 0) {
        Inputs.reserve(Pattern->consumer_ports.size());
        for (auto& port : Pattern->consumer_ports) {
            Inputs.emplace_back(std::make_shared<DFGPin>( &port, this));
        }

        max_output_name_length = 0;

        Outputs.reserve(Pattern->producer_ports.size());
        for (auto& port : Pattern->producer_ports) {
            max_output_name_length = std::max(max_output_name_length, port.getName().length());
            Outputs.emplace_back(std::make_shared<DFGPin>(&port, this));
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
}