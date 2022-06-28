/** Copyright (C) 2022  Frieder Pankratz <frieder.pankratz@gmail.com> **/

#ifndef TRAACT_GUI_SRC_TRAACT_GUI_DATAFLOWFILE_H_
#define TRAACT_GUI_SRC_TRAACT_GUI_DATAFLOWFILE_H_

#include <string>
#include <util/fileutil.h>
#include <external/imgui-node-editor/imgui_node_editor.h>
#include <traact/traact.h>
#include <traact/facade/Facade.h>
#include <traact_gui/editor/PatternGraphEditor.h>
#include "SelectedTraactElement.h"
#include "traact_gui/debug_run/DebugRun.h"
namespace traact::gui {

 struct DataflowFile : public std::enable_shared_from_this<DataflowFile> {
        explicit DataflowFile(std::string name,
                              SelectedTraactElement &t_selected_traact_element);
        explicit DataflowFile(fs::path file,
                              SelectedTraactElement &t_selected_traact_element);
        explicit DataflowFile(nlohmann::json graph,
                              SelectedTraactElement &t_selected_traact_element);
        ~DataflowFile();


        fs::path filepath;
        SelectedTraactElement& selected_traact_element;

        bool        open;       // Set when open (we keep an array of all available documents to simplify demo code!)
        bool        openPrev;   // Copy of Open from last update.
        bool        dirty;      // Set when the document has been modified
        bool        wantClose;  // Set when the document
        bool        wantSave;
        //bool        selected{false};

        void doOpen()       { open = true; }
        void doQueueClose() { wantClose = true; }
        void doQueueSave() { wantSave = true; }
        void doForceClose() { open = false; dirty = false; }
        void doSave();

        void draw();
        void drawContextMenu();
        void drawSrgPanel();
        void drawDfgPanel();
        void setDropTarget();

        [[nodiscard]] std::string toDataString() const;

        [[nodiscard]] const char* getName() const;
        [[nodiscard]] const std::string& getNameString() const;

        void startDataflow();

        ax::NodeEditor::EditorContext* context_srg_ = nullptr;
        ax::NodeEditor::EditorContext* context_dfg_ = nullptr;

        std::shared_ptr<traact::facade::Facade> facade_;
        //DefaultInstanceGraphPtr component_graph_;
        editor::PatternGraphEditor graph_editor_;

        ax::NodeEditor::NodeId contextNodeId{0};
        ax::NodeEditor::LinkId contextLinkId{0};
        ax::NodeEditor::PinId  contextPinId{0};
        bool createNewNode{false};
        editor::DFGPin::Ptr newNodeLinkPin{nullptr};
        editor::DFGPin::Ptr newLinkPin{nullptr};



        //srg
        ax::NodeEditor::NodeId draggedNodeId{ax::NodeEditor::NodeId::Invalid};
        //----

        void buildNodes();
        void layoutDfgNodes();
        void layoutDfgNodesFromSinks();
        void layoutDfgNodesNodeSoup();
        void layoutSrgNodes();
        void layoutSrgNode(editor::EditorPattern::Ptr pattern);



        ImVec2 CurrentEditorSize;
        ImVec2 CurrentEditorPos;

        bool canUndo() const;
        bool canRedo() const;

        void saveState();
        void undo();
        void redo();


    private:

        std::stack<nlohmann::json> undo_buffer_;
        std::stack<nlohmann::json> redo_buffer_;

    };
}

#endif //TRAACT_GUI_SRC_TRAACT_GUI_DATAFLOWFILE_H_

