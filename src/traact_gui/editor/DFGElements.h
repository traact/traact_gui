/** Copyright (C) 2022  Frieder Pankratz <frieder.pankratz@gmail.com> **/

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
