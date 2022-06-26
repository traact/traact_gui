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

#ifndef TRAACT_GUI_SRC_TRAACT_GUI_EDITOR_DFGELEMENTS_H_
#define TRAACT_GUI_SRC_TRAACT_GUI_EDITOR_DFGELEMENTS_H_

#include <spdlog/spdlog.h>

#include <utility>
#include <external/imgui-node-editor/imgui_node_editor.h>
#include <traact/pattern/instance/PatternInstance.h>

namespace traact::gui::editor {

    struct DFGLink
    {
        typedef typename std::shared_ptr<DFGLink> Ptr;
        ax::NodeEditor::LinkId ID;

        ax::NodeEditor::PinId StartPinID;
        ax::NodeEditor::PinId EndPinID;

        ImColor Color;

        DFGLink(ax::NodeEditor::PinId startPinId, ax::NodeEditor::PinId endPinId);
    };



    struct DFGNode;

    struct DFGPin {
        typedef typename std::shared_ptr<DFGPin> Ptr;
        ax::NodeEditor::PinId ID;
        DFGNode *ParentNode;

        pattern::instance::PortInstance::ConstPtr TraactPort;

        DFGPin(const pattern::instance::PortInstance * port, DFGNode *node);
    };

    struct DFGNode {
        typedef typename std::shared_ptr<DFGNode> Ptr;
        ax::NodeEditor::NodeId ID;
        pattern::instance::PatternInstance::Ptr Pattern;

        std::vector <DFGPin::Ptr> Inputs;
        std::vector <DFGPin::Ptr> Outputs;

        ImVec2 Position;

        void SavePosition();
        void RestorePosition();
        const std::string& getName() const;

        ImVec2 Size;
        std::size_t max_output_name_length;

        std::size_t OutputWeight;
        std::size_t InputWeight;

        DFGNode(const pattern::instance::PatternInstance::Ptr& pattern);
        void UpdateOutputWeight();
        void UpdateInputWeight();
    };
}

#endif //TRAACT_GUI_SRC_TRAACT_GUI_EDITOR_DFGELEMENTS_H_
