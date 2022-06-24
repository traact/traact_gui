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

#ifndef TRAACTMULTI_DATAFLOWFILE_H
#define TRAACTMULTI_DATAFLOWFILE_H

#include <string>
#include <util/fileutil.h>
#include <external/imgui-node-editor/imgui_node_editor.h>
#include <traact/traact.h>
#include <traact/facade/Facade.h>
#include <traact/util/RingBuffer.h>
#include <traact_gui/editor/PatternGraphEditor.h>

namespace traact::gui {

    struct DataflowFile {
        explicit DataflowFile(std::string name);
        explicit DataflowFile(fs::path file);
        explicit DataflowFile(nlohmann::json graph);
        ~DataflowFile();


        fs::path filepath;
        bool        open;       // Set when open (we keep an array of all available documents to simplify demo code!)
        bool        openPrev;   // Copy of Open from last update.
        bool        dirty;      // Set when the document has been modified
        bool        wantClose;  // Set when the document
        bool        wantSave;
        //bool        selected{false};

        void DoOpen()       { open = true; }
        void DoQueueClose() { wantClose = true; }
        void DoQueueSave() { wantSave = true; }
        void DoForceClose() { open = false; dirty = false; }
        void DoSave();

        void Draw(int width, int height);
        void DrawContextMenu();
        void DrawSrgPanel(int width, int height);
        void DrawDfgPanel(int width, int height);
        void SetDropTarget();

        [[nodiscard]] std::string ToDataString() const;

        [[nodiscard]] const char* GetName() const;
        [[nodiscard]] const std::string& GetNameString() const;
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

        void BuildNodes();
        void LayoutDFGNodes();
        void LayoutDFGNodesFromSinks();
        void LayoutDFGNodesNodeSoup();
        void LayoutSRGNodes();
        void LayoutSRGNode(editor::EditorPattern::Ptr pattern);



        ImVec2 CurrentEditorSize;
        ImVec2 CurrentEditorPos;

        bool CanUndo() const;
        bool CanRedo() const;

        void SaveState();
        void Undo();
        void Redo();


    private:

        std::stack<nlohmann::json> undo_buffer_;
        std::stack<nlohmann::json> redo_buffer_;

    };
}

#endif //TRAACTMULTI_DATAFLOWFILE_H

