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

#include "DataflowFile.h"
#include "traact_gui/editor/EditorUtils.h"
#include <spdlog/spdlog.h>
#include <traact_gui/ImGuiUtils.h>
#include <traact/facade/DefaultFacade.h>
#include <traact/serialization/JsonGraphInstance.h>
#include <traact/pattern/instance/GraphInstance.h>
#include <external/imgui-node-editor/utilities/widgets.h>
#include "traact_gui/ImGuiUtils.h"
#include "traact_gui/editor/JsonEditorSerialization.h"

#include <nodesoup.hpp>
#include <unordered_map>

namespace ed = ax::NodeEditor;


static inline ImRect ImGui_GetItemRect()
{
    return ImRect(ImGui::GetItemRectMin(), ImGui::GetItemRectMax());
}

namespace traact::gui {

    DataflowFile::DataflowFile(std::string name) {
        open = openPrev = true;
        dirty = true;
        wantSave = wantClose = false;
        context_srg_ = ed::CreateEditor();
        context_dfg_ = ed::CreateEditor();
        graph_editor_.Graph = std::make_shared<DefaultInstanceGraph>(std::move(name));
        facade_ = std::make_shared<facade::DefaultFacade>();
        BuildNodes();
    }

    DataflowFile::DataflowFile(fs::path file) : filepath(file){
        open = openPrev = true;
        dirty = false;
        wantSave = wantClose = false;
        context_srg_ = ed::CreateEditor();
        context_dfg_ = ed::CreateEditor();
        facade_ = std::make_shared<facade::DefaultFacade>();


        nlohmann::json json_graph;
        std::ifstream graph_file;
        graph_file.open(filepath.string());
        graph_file >> json_graph;
        graph_file.close();

        ns::from_json(json_graph, graph_editor_);

        BuildNodes();
    }

    DataflowFile::DataflowFile(nlohmann::json graph) {
        open = openPrev = true;
        dirty = false;
        wantSave = wantClose = false;
        context_srg_ = ed::CreateEditor();
        context_dfg_ = ed::CreateEditor();
        facade_ = std::make_shared<facade::DefaultFacade>();


        ns::from_json(graph, graph_editor_);
        graph_editor_.Graph->name = graph_editor_.Graph->name + " (copy)";
        BuildNodes();
    }

    DataflowFile::~DataflowFile() {
        ed::DestroyEditor(context_srg_);
        ed::DestroyEditor(context_dfg_);
    }

    void DataflowFile::Draw(int width, int height) {
        ImGui::PushID(this);


        if (ImGui::BeginTabBar("##srg_dfg_tabs")) {
            if(ImGui::BeginTabItem("Spatial Relationship Graph##srg_tab")){
                DrawSrgPanel(width, height);
                ImGui::EndTabItem();
            }
            if(ImGui::BeginTabItem("Dataflow Graph##dft_tab")){
                DrawDfgPanel(width, height);
                ImGui::EndTabItem();
            }

            ImGui::EndTabBar();
        }

//        static float srgHeight  = 600.0f;
//        static float dfgHeight = 400.0f;
//
//        Splitter("##SplitterRight",false, 4.0f, &srgHeight, &dfgHeight, 50.0f, 50.0f);
//        ImGui::BeginChild("right", ImVec2(width, srgHeight+dfgHeight+4));
//
//        ImGui::BeginChild("right up", ImVec2(width, srgHeight+4));
//
//        ImGui::EndChild();
//
//        ImGui::BeginChild("right down");
//        DrawDfgPanel(dataflow, width, dfgHeight);
//        ImGui::EndChild();
//        ImGui::EndChild();

       // ImGui::TextWrapped("fsdfsdf  sadfased fsdfsdf  sadfased fsdfsdf  sadfased fsdfsdf  sadfased fsdfsdf  sadfased fsdfsdf  sadfased fsdfsdf  sadfased fsdfsdf  sadfased ");


        ImGui::PopID();
    }

    void DataflowFile::DrawContextMenu() {
        if (!ImGui::BeginPopupContextItem())
            return;

        char buf[256];
        sprintf(buf, "Save %s", GetName());
        if (ImGui::MenuItem(buf, "CTRL+S", false, open))
            DoQueueSave();
        if (ImGui::MenuItem("Close", "CTRL+W", false, open))
            DoQueueClose();
        ImGui::EndPopup();
    }

    void DataflowFile::DrawSrgPanel(int width, int height) {
        ed::SetCurrentEditor(context_srg_);
        ed::Begin("srg_editor");

        CurrentEditorSize = ImGui::GetWindowSize();
        CurrentEditorPos = ImGui::GetWindowPos();

        const static ImColor input_color(255,0,0,255);
        const static ImColor output_color(0,255,0,255);
        const float TEXT_BASE_WIDTH = ImGui::CalcTextSize("A").x;
        const float TEXT_BASE_HEIGHT = ImGui::GetTextLineHeightWithSpacing();

        const float pin_size(20);
        const ImVec2 pin_vec(pin_size,pin_size);
        const ImVec2 half_pin_vec(pin_size/2,pin_size/2);
        ImVec2 group_size(200,200);

        auto draw_srg_node = [this](ed::NodeId id, ed::PinId outputPin, ed::PinId inputPin, const char* name) -> void {
            ed::PushStyleVar(ed::StyleVar_NodeRounding, 10);
            ed::PushStyleVar(ed::StyleVar_SourceDirection, ImVec2(0.0f, 0.0f));
            ed::PushStyleVar(ed::StyleVar_TargetDirection, ImVec2(0.0f, 0.0f));


            ed::BeginNode(id);
            ImGui::BeginGroup();
            ImGui::Dummy(ImVec2(80, 0));
            ImGui::Text("%s", name);
            ImGui::EndGroup();

            auto input_rect = ImGui_GetItemRect();
            ImVec2 padding(10, 10);
            input_rect = ImRect(input_rect.GetTL() - padding, input_rect.GetBR() + padding);


            ed::PushStyleVar(ed::StyleVar_PinArrowSize, 10.0f);
            ed::PushStyleVar(ed::StyleVar_PinArrowWidth, 10.0f);
            ed::BeginPin(inputPin, ed::PinKind::Input);
            ImGui::Dummy(ImVec2(1, 1));
            ed::PinPivotRect(input_rect.GetTL(), input_rect.GetBR());
            ed::EndPin();

            ed::PopStyleVar(2);

            ImGui::SameLine();

            ed::BeginPin(outputPin, ed::PinKind::Output);
            ImGui::Dummy(ImVec2(1, 1));
            ed::PinPivotRect(input_rect.GetTL(), input_rect.GetBR());
            //ed::PinPivotSize(node_size);
            ed::EndPin();



            ed::EndNode();

            ed::PopStyleVar(3);


            if (ImGui::IsMouseDragging(ImGuiMouseButton_Left)) {
                //spdlog::info("IsMouseDragging {0}", local_node->CoordinateSystem->name);
            }

            if (ImGui::IsItemClicked(ImGuiMouseButton_Left)) {
                draggedNodeId = id;
            }


            if(ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenBlockedByActiveItem) &&
                draggedNodeId != id &&
                draggedNodeId != ed::NodeId::Invalid) {


                auto node_error = graph_editor_.CanMergeNodes(draggedNodeId, id);
                if(node_error.has_value()){

                    ShowLabel(node_error.value().c_str(), ImColor(45, 32, 32, 180), true);
                } else {
                    ShowLabel("Merge Nodes", ImColor(32, 45, 32, 180), true);
                    if(ImGui::IsMouseReleased(ImGuiMouseButton_Left)) {
                        SaveState();
                        auto new_merge_node = graph_editor_.MergeNodes(draggedNodeId, id);
                        if(new_merge_node){
                            ed::SetNodePosition(new_merge_node->ID, new_merge_node->Position);
                        }
                    }

                }




            }

        };

        for (auto& pattern : graph_editor_.Patterns) {
            for(auto& local_node : pattern->SrgNodes) {
                if (!local_node->IsVisible())
                    continue;

                draw_srg_node(local_node->ID, local_node->GetSourceID(), local_node->GetTargetID(), local_node->CoordinateSystem->name.c_str());
                local_node->SavePosition();

            }
        }

        // merged nodes
        auto srg_merged_node_copy = graph_editor_.SrgMergedNodes;
        for (auto& local_node : srg_merged_node_copy) {
            draw_srg_node(local_node->ID, local_node->OutputPinID, local_node->InputPinID, local_node->Name.c_str());
            local_node->SavePosition();
        }

        // srg edges
        for (auto& pattern : graph_editor_.Patterns) {
            for (auto &link: pattern->SrgConnections) {

                if (!link->IsVisible())
                    continue;
                auto link_color = link->IsOutput ? output_color : input_color;
                ed::Link(link->ID, link->GetSourceID(), link->GetTargetID(), link_color, 2.0f);

            }
        }

        // merged edges
        auto srg_merged_edge_copy = graph_editor_.SrgMergedEdges;
        for (auto& local_edge : srg_merged_edge_copy) {
            ed::Link(local_edge->ID, local_edge->OutputPinID, local_edge->InputPinID, output_color, 2.0f);
        }

        // edit connections
        if (!createNewNode)
        {






        }




        if(ed::IsBackgroundClicked() || ImGui::IsMouseReleased(ImGuiMouseButton_Left)){
            draggedNodeId = ed::NodeId::Invalid;
        }

        // context menu
        auto openPopupPosition = ImGui::GetMousePos();
        ed::Suspend();
        if (ed::ShowNodeContextMenu(&contextNodeId))
            ImGui::OpenPopup("SRG Node Context Menu");
        else if (ed::ShowPinContextMenu(&contextPinId))
            ImGui::OpenPopup("SRG Pin Context Menu");
        else if (ed::ShowLinkContextMenu(&contextLinkId))
            ImGui::OpenPopup("SRG Link Context Menu");
        else if (ed::ShowBackgroundContextMenu())
        {
            ImGui::OpenPopup("SRG Context Menu");
        }
        ed::Resume();

        ed::Suspend();
        ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(8, 8));
        if (ImGui::BeginPopup("SRG Node Context Menu"))
        {
            ImGui::TextUnformatted("Node Context Menu");
            ImGui::Separator();
            auto merge_node = graph_editor_.FindSrgMergeNode(contextNodeId);
            if (merge_node)
            {
                ImGui::Text("Name: %s", merge_node->Name.c_str());
                ImGui::Separator();
                for(const auto& child : merge_node->Children) {
                    ImGui::Text("%s", child->GetName().c_str());
                }
                ImGui::Separator();
                if (ImGui::MenuItem("Split Nodes")){
                    graph_editor_.SplitNode(contextNodeId);
                }
            }
            else {
                auto srg_node = graph_editor_.FindSrgNode(contextNodeId);
                if(srg_node) {
                    ImGui::Text("%s", srg_node->GetName().c_str());
                }
                else
                    ImGui::Text("Unknown node: %p", contextNodeId.AsPointer());
            }

            ImGui::Separator();
            if (ImGui::MenuItem("Delete")){
                //ed::DeleteNode(contextNodeId);
            }

            ImGui::EndPopup();
        }

        if (ImGui::BeginPopup("SRG Pin Context Menu"))
        {
//            auto pin = graph_editor_.FindPin(contextPinId);
//
//            ImGui::TextUnformatted("Pin Context Menu");
//            ImGui::Separator();
//            if (pin)
//            {
//                ImGui::Text("Name: %s", pin->TraactPort->getName().c_str());
//                ImGui::Text("Type: %s", pin->TraactPort->getDataType().c_str());
//            }
//            else
//                ImGui::Text("Unknown pin: %p", contextPinId.AsPointer());
//
            ImGui::EndPopup();
        }


        if (ImGui::BeginPopup("SRG Link Context Menu"))
        {
//            auto link = graph_editor_.FindLink(contextLinkId);
//
//            ImGui::TextUnformatted("Link Context Menu");
//            ImGui::Separator();
//            if (link)
//            {
//                auto start_pin = graph_editor_.FindPin(link->StartPinID);
//                auto end_pin = graph_editor_.FindPin(link->EndPinID);
//                ImGui::Text("Type: %s", start_pin->TraactPort->getDataType().c_str());
//                ImGui::Text("From: %s:%s", start_pin->TraactPort->getID().first.c_str(),start_pin->TraactPort->getID().second.c_str());
//                ImGui::Text("To:   %s:%s", end_pin->TraactPort->getID().first.c_str(),end_pin->TraactPort->getID().second.c_str());
//            }
//            else
//                ImGui::Text("Unknown link: %p", contextLinkId.AsPointer());
//            ImGui::Separator();
//            if (ImGui::MenuItem("Delete"))
//                ed::DeleteLink(contextLinkId);
            ImGui::EndPopup();
        }

        if (ImGui::BeginPopup("SRG Context Menu"))
        {
//            auto newNodePostion = openPopupPosition;
//
//            Node* node = nullptr;
//
            if (ImGui::MenuItem("Layout Nodes")){
                this->LayoutSRGNodes();
            }

            ImGui::EndPopup();
        }

        ImGui::PopStyleVar();
        ed::Resume();



        ed::End();

        if (ImGui::BeginDragDropTarget())
        {
            if (const ImGuiPayload* payload = ImGui::AcceptDragDropPayload("NEW_PATTERN_DRAGDROP"))
            {
                std::string new_pattern(static_cast<char*>(payload->Data));

                auto node_pos = GetCanvasMousePosition();

                auto pattern = facade_->instantiatePattern(new_pattern);
                auto new_editor_pattern = graph_editor_.CreatePatternInstance(pattern);
                LayoutSRGNode(new_editor_pattern);

                //ed::SetNodePosition(new_node_id, node_pos);
            }
            ImGui::EndDragDropTarget();
        }

    }

    void DataflowFile::DrawDfgPanel(int width, int height) {
        ed::SetCurrentEditor(context_dfg_);
        ed::Begin("dfg_editor");


        const static ImColor input_color(255,0,0,255);
        const static ImColor output_color(0,255,0,255);
        const float TEXT_BASE_WIDTH = ImGui::CalcTextSize("A").x;
        const float TEXT_BASE_HEIGHT = ImGui::GetTextLineHeightWithSpacing();

        const float pin_size(20);
        const ImVec2 pin_vec(pin_size,pin_size);
        const ImVec2 half_pin_vec(pin_size/2,pin_size/2);
        ImVec2 group_size(200,200);


        for (auto& pattern : graph_editor_.Patterns) {
            auto& local_node = pattern->DfgNode;

            ed::BeginNode(local_node->ID);
            ImGui::Text("%s", local_node->Pattern->instance_id.c_str());

            ImGui::BeginGroup();
            ImRect inputsRect;


            for (auto& input : local_node->Inputs){
                ed::BeginPin(input->ID, ed::PinKind::Input);

                ax::Widgets::Icon(pin_vec, ax::Drawing::IconType::Circle, false, input_color, ImColor(32, 32, 32, 0));
                inputsRect = ImGui_GetItemRect();
                ed::PinPivotRect(inputsRect.GetTL()+half_pin_vec, inputsRect.GetTL()+half_pin_vec);
                ImGui::SameLine();
                ImGui::Text("%s",input->TraactPort->getName().c_str());
                ed::EndPin();


            }
            ImGui::EndGroup();

            ImGui::SameLine();

            ImGui::BeginGroup();
            for (auto& output : local_node->Outputs){
                ed::BeginPin(output->ID, ed::PinKind::Output);

                ImGui::Text("%s",output->TraactPort->getName().c_str());
                float offset = TEXT_BASE_WIDTH * static_cast<float>(local_node->max_output_name_length);
                ImGui::SameLine(offset);
                ax::Widgets::Icon(pin_vec, ax::Drawing::IconType::Circle, false, output_color, ImColor(32, 32, 32, 0));
                inputsRect = ImGui_GetItemRect();
                ed::PinPivotRect(inputsRect.GetTL()+half_pin_vec, inputsRect.GetTL()+half_pin_vec);
                ed::EndPin();
            }
            ImGui::EndGroup();

            ed::EndNode();

            local_node->SavePosition();

        }


        for (auto& link : graph_editor_.DfgConnections)
            ed::Link(link->ID, link->StartPinID, link->EndPinID, link->Color, 2.0f);

        // edit connections
        if (!createNewNode)
        {
            if (ed::BeginCreate(ImColor(255, 255, 255), 2.0f))
            {
                

                ed::PinId startPinId = 0, endPinId = 0;
                if (ed::QueryNewLink(&startPinId, &endPinId))
                {
                    auto startPin = graph_editor_.FindPin(startPinId);
                    auto endPin   = graph_editor_.FindPin(endPinId);

                    newLinkPin = startPin ? startPin : endPin;

                    if (startPin->TraactPort->port.porttype == pattern::PortType::Consumer)
                    {
                        std::swap(startPin, endPin);
                        std::swap(startPinId, endPinId);
                    }

                    if (startPin && endPin)
                    {
                        if (endPin == startPin)
                        {
                            ed::RejectNewItem(ImColor(255, 0, 0), 2.0f);
                        }

                        std::optional<std::string> link_error = graph_editor_.CanCreateLink(startPin, endPin);
                        if(link_error.has_value()){
                            ShowLabel(link_error.value().c_str(), ImColor(45, 32, 32, 180));
                            ed::RejectNewItem(ImColor(255, 128, 128), 1.0f);
                        } else {
                            ShowLabel("+ Create Link", ImColor(32, 45, 32, 180));
                            if (ed::AcceptNewItem(ImColor(128, 255, 128), 4.0f))
                            {
                                SaveState();
                                graph_editor_.ConnectPins(startPinId, endPinId);


                            }
                        }


//                        else if (endPin->TraactPort->port.datatype != startPin->TraactPort->port.datatype)
//                        {
//                            ShowLabel("x Incompatible Pin datatype", ImColor(45, 32, 32, 180));
//                            ed::RejectNewItem(ImColor(255, 0, 0), 2.0f);
//                        }
//                        else if (endPin->ParentNode == startPin->ParentNode)
//                        {
//                            ShowLabel("x Cannot connect to self", ImColor(45, 32, 32, 180));
//                            ed::RejectNewItem(ImColor(255, 0, 0), 1.0f);
//                        }
//                        else if (endPin->TraactPort->port.porttype == startPin->TraactPort->port.porttype)
//                        {
//                            ShowLabel("x Incompatible Pin Type, can only connect input to output", ImColor(45, 32, 32, 180));
//                            ed::RejectNewItem(ImColor(255, 128, 128), 1.0f);
//                        }
//                        else
//                        {
//                            std::optional<std::string> link_error = graph_editor_.CanCreateLink(startPin, endPin);
//                            if(link_error.has_value()){
//                                ShowLabel(link_error.value().c_str(), ImColor(45, 32, 32, 180));
//                                ed::RejectNewItem(ImColor(255, 128, 128), 1.0f);
//                            }
//                            ShowLabel("+ Create Link", ImColor(32, 45, 32, 180));
//                            if (ed::AcceptNewItem(ImColor(128, 255, 128), 4.0f))
//                            {
//                                graph_editor_.ConnectPins(startPinId, endPinId);
//                                dirty = true;
//                            }
//                        }
                    }
                }

            }
            else
                newLinkPin = nullptr;

            ed::EndCreate();

            if (ed::BeginDelete())
            {
                ed::LinkId linkId = 0;
                while (ed::QueryDeletedLink(&linkId))
                {
                    if (ed::AcceptDeletedItem())
                    {
                        SaveState();
                        graph_editor_.DisconnectPin(linkId);
                    }
                }

                ed::NodeId nodeId = 0;
                while (ed::QueryDeletedNode(&nodeId))
                {
                    if (ed::AcceptDeletedItem())
                    {
                        graph_editor_.DeleteNode(nodeId);

                    }
                }
            }
            ed::EndDelete();
        }

        // context menu
        auto openPopupPosition = ImGui::GetMousePos();
        ed::Suspend();
        if (ed::ShowNodeContextMenu(&contextNodeId))
            ImGui::OpenPopup("Node Context Menu");
        else if (ed::ShowPinContextMenu(&contextPinId))
            ImGui::OpenPopup("Pin Context Menu");
        else if (ed::ShowLinkContextMenu(&contextLinkId))
            ImGui::OpenPopup("Link Context Menu");
        else if (ed::ShowBackgroundContextMenu())
        {
            ImGui::OpenPopup("Create New Node");
            newNodeLinkPin = nullptr;
        }
        ed::Resume();

        ed::Suspend();
        ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(8, 8));
        if (ImGui::BeginPopup("Node Context Menu"))
        {
            auto node = graph_editor_.FindNode(contextNodeId);

            ImGui::TextUnformatted("Node Context Menu");
            ImGui::Separator();
            if (node)
            {
                ImGui::Text("ID: %s", node->Pattern->instance_id.c_str());
                ImGui::Text("Type: %s", node->Pattern->local_pattern.name.c_str());
                ImGui::Text("Inputs: %d", (int)node->Inputs.size());
                ImGui::Text("Outputs: %d", (int)node->Outputs.size());
            }
            else
                ImGui::Text("Unknown node: %p", contextNodeId.AsPointer());
            ImGui::Separator();
            if (ImGui::MenuItem("Delete"))
                ed::DeleteNode(contextNodeId);
            ImGui::EndPopup();
        }

        if (ImGui::BeginPopup("Pin Context Menu"))
        {
            auto pin = graph_editor_.FindPin(contextPinId);

            ImGui::TextUnformatted("Pin Context Menu");
            ImGui::Separator();
            if (pin)
            {
                ImGui::Text("Name: %s", pin->TraactPort->getName().c_str());
                ImGui::Text("Type: %s", pin->TraactPort->getDataType().c_str());
            }
            else
                ImGui::Text("Unknown pin: %p", contextPinId.AsPointer());

            ImGui::EndPopup();
        }


        if (ImGui::BeginPopup("Link Context Menu"))
        {
            auto link = graph_editor_.FindLink(contextLinkId);

            ImGui::TextUnformatted("Link Context Menu");
            ImGui::Separator();
            if (link)
            {
                auto start_pin = graph_editor_.FindPin(link->StartPinID);
                auto end_pin = graph_editor_.FindPin(link->EndPinID);
                ImGui::Text("Type: %s", start_pin->TraactPort->getDataType().c_str());
                ImGui::Text("From: %s:%s", start_pin->TraactPort->getID().first.c_str(),start_pin->TraactPort->getID().second.c_str());
                ImGui::Text("To:   %s:%s", end_pin->TraactPort->getID().first.c_str(),end_pin->TraactPort->getID().second.c_str());
            }
            else
                ImGui::Text("Unknown link: %p", contextLinkId.AsPointer());
            ImGui::Separator();
            if (ImGui::MenuItem("Delete"))
                ed::DeleteLink(contextLinkId);
            ImGui::EndPopup();
        }

        if (ImGui::BeginPopup("Create New Node"))
        {
            auto newNodePostion = openPopupPosition;
            //ImGui::SetCursorScreenPos(ImGui::GetMousePosOnOpeningCurrentPopup());

            //auto drawList = ImGui::GetWindowDrawList();
            //drawList->AddCircleFilled(ImGui::GetMousePosOnOpeningCurrentPopup(), 10.0f, 0xFFFF00FF);

            editor::DFGNode* node = nullptr;

            ImGui::Separator();
            if (ImGui::MenuItem("Layout Nodes from Sinks"))
                LayoutDFGNodesFromSinks();
            if (ImGui::MenuItem("Layout Nodes from Sources"))
                LayoutDFGNodes();
            if (ImGui::MenuItem("Layout Nodes using NodeSoup"))
                LayoutDFGNodesNodeSoup();


//            ImGui::Separator();
//            if (ImGui::MenuItem("Comment"))
//                node = SpawnComment();




            ImGui::EndPopup();
        }
        else
            createNewNode = false;
        ImGui::PopStyleVar();
        ed::Resume();



        ed::End();

        SetDropTarget();
    }

    void DataflowFile::SetDropTarget() {
        if (ImGui::BeginDragDropTarget())
        {
            if (const ImGuiPayload* payload = ImGui::AcceptDragDropPayload("NEW_PATTERN_DRAGDROP"))
            {
                std::string new_pattern(static_cast<char*>(payload->Data));
                auto mouse_pos = ImGui::GetMousePos();
                auto node_pos = ed::ScreenToCanvas(mouse_pos);

                auto pattern = facade_->instantiatePattern(new_pattern);
                auto new_editor_pattern = graph_editor_.CreatePatternInstance(pattern);
                ed::SetNodePosition(new_editor_pattern->DfgNode->ID, node_pos);
            }
            ImGui::EndDragDropTarget();
        }
    }

    const char *DataflowFile::GetName() const{
        return GetNameString().c_str();
    }

    const std::string &DataflowFile::GetNameString() const{
        return graph_editor_.Graph->name;
    }

    void DataflowFile::DoSave() {
        try{

            std::ofstream graph_file;
            graph_file.open(filepath.string());

            graph_file << ToDataString();
            graph_file.close();

            dirty = false;
        } catch (...){
            spdlog::error("error saving dataflow file {0}", filepath.string());
        }


    }

    std::string DataflowFile::ToDataString() const {
        nlohmann::json json_graph;
        ns::to_json(json_graph, graph_editor_);
        return json_graph.dump(4);
    }

    void DataflowFile::BuildNodes() {

        graph_editor_.CreateNodes();
        graph_editor_.CreateConnections();

    }

    void DataflowFile::LayoutDFGNodes() {
        //std::vector<ImVec2> node_sizes;
        std::vector<std::vector<editor::DFGNode::Ptr> > node_table;

        auto& nodes = graph_editor_.Patterns;
        auto node_count = nodes.size();

        std::vector<editor::DFGNode::Ptr> unsorted_nodes;
        unsorted_nodes.reserve(node_count);

        // sort out sources and not connected components for the first two columns
        // 0: source (no input), 1: not connected input, rest unsorted
        node_table.resize(2);
        for (std::size_t i = 0; i < node_count; ++i) {
            auto& node = nodes[i]->DfgNode;
            node->UpdateOutputWeight();
            if(node->Inputs.empty()) {
                node_table[0].emplace_back(node);
            } else {

                bool inputs_connected = true;

                for (const auto& input_pin : node->Inputs) {
                    inputs_connected = inputs_connected && input_pin->TraactPort->IsConnected();
                }

                if(!inputs_connected){
                    node_table[1].emplace_back(node);
                } else {
                    unsorted_nodes.emplace_back(node);
                }
            }
        }

        auto sort_by_output =  [](const editor::DFGNode::Ptr& a, const editor::DFGNode::Ptr& b) -> bool
        {
            return a->OutputWeight > b->OutputWeight;
        };

        sort(node_table[0].begin(), node_table[0].end(), sort_by_output);
        sort(node_table[1].begin(), node_table[1].end(), sort_by_output);

        std::size_t current_column = 2;
        auto is_prev_pattern = [&node_table, current_column](const pattern::instance::PatternInstance::Ptr& a) -> bool
        {
            for (const auto& column : node_table) {
                auto result = std::find_if(column.cbegin(), column.cend(), [&a](const editor::DFGNode::Ptr b) -> bool
                {
                    return a == b->Pattern;
                });
                if(result != column.cend())
                    return true;
            }

            return false;
        };

        while(!unsorted_nodes.empty()){
            std::vector<editor::DFGNode::Ptr> next_column;
            std::vector<editor::DFGNode::Ptr> current_nodes;
            current_nodes.reserve(node_count);
            for (editor::DFGNode::Ptr& node : unsorted_nodes) {
                current_nodes.emplace_back(node);
            }

            for (editor::DFGNode::Ptr& node : current_nodes) {
                bool only_connected_to_prev = true;
                for (const auto& input_pin : node->Inputs) {
                    if(!input_pin->TraactPort->IsConnected())
                        continue;
                    auto pattern_port = input_pin->TraactPort->connected_to;
                    auto source_pattern = graph_editor_.Graph->getPattern(pattern_port.first);
                    if(!is_prev_pattern(source_pattern)){
                        only_connected_to_prev = false;
                        break;
                    }
                }
                if(only_connected_to_prev) {
                    next_column.push_back(node);
                    auto remove_it = std::remove(unsorted_nodes.begin(),unsorted_nodes.end(), node);
                    unsorted_nodes.erase(remove_it, unsorted_nodes.end());
                }
            }



            sort(next_column.begin(), next_column.end(), sort_by_output);
            node_table.emplace_back(next_column);
            current_column++;
        }

        ImVec2 node_position(0,0);
        float padding = 10;
        float node_distance = 50;
        for (auto& column : node_table) {
            float max_width = 0;
            for (auto cell : column) {
                ed::SetNodePosition(cell->ID, node_position);
                auto node_size = ed::GetNodeSize(cell->ID);
                max_width = std::max(max_width, node_size.x);
                node_position.y += node_size.y + padding;
            }
            node_position.x += max_width + node_distance;
            node_position.y = 0;
        }

    }

    void DataflowFile::LayoutSRGNodes() {
        auto screen_size = ed::GetScreenSize();
        auto width = static_cast<unsigned int>(screen_size.x*ed::GetCurrentZoom());
        auto height = static_cast<unsigned int>(screen_size.y*ed::GetCurrentZoom());

        auto layout_center = ed::ScreenToCanvas(CurrentEditorPos + (CurrentEditorSize / 2));//GetCanvasMousePosition();
        double k = 15;
        double energy_threshold = 1e-2;
        int iters_count = 300;

        nodesoup::adj_list_t g;
        std::unordered_map<void*, nodesoup::vertex_id_t> nodeId_vertexId;
        std::unordered_map<nodesoup::vertex_id_t, editor::SRGNode::Ptr> srg_nodes;
        auto name_to_vertex_id = [&g, &nodeId_vertexId, &srg_nodes](editor::SRGNode::Ptr node) -> nodesoup::vertex_id_t {

            auto name = node->ID.AsPointer();
            nodesoup::vertex_id_t v_id;
            auto it = nodeId_vertexId.find(name);
            if (it != nodeId_vertexId.end()) {
                return (*it).second;
            }

            v_id = g.size();
            nodeId_vertexId.insert({name, v_id });
            srg_nodes.emplace(v_id, node);
            g.resize(v_id + 1);
            return v_id;
        };

        for (const auto& pattern : graph_editor_.Patterns)
        for (const auto& edge : pattern->SrgConnections) {
            if(!edge->IsVisible())
                continue;

            // add vertex if new
            nodesoup::vertex_id_t v_id = name_to_vertex_id(edge->SourceNode);

            // add adjacent vertex if new
            nodesoup::vertex_id_t adj_id = name_to_vertex_id(edge->TargetNode);

            // add edge if new
            if (find(g[v_id].begin(), g[v_id].end(), adj_id) == g[v_id].end()) {
                g[v_id].push_back(adj_id);
                g[adj_id].push_back(v_id);
            }

        }

        std::vector<nodesoup::Point2D> positions = nodesoup::fruchterman_reingold(g, width, height, iters_count, k,
                                                                                  nullptr);


        for (nodesoup::vertex_id_t v_id = 0; v_id < g.size(); v_id++) {
            positions[v_id].x += layout_center.x;
            positions[v_id].y += layout_center.y;
        }
        for (nodesoup::vertex_id_t v_id = 0; v_id < g.size(); v_id++) {
            nodesoup::Point2D v_pos = positions[v_id];
            auto node = srg_nodes.at(v_id);

            ed::SetNodePosition(node->ID, ImVec2(v_pos.x, v_pos.y));
        }
    }

    void DataflowFile::LayoutDFGNodesNodeSoup() {
        auto screen_size = ed::GetScreenSize();
        auto width = static_cast<unsigned int>(screen_size.x*ed::GetCurrentZoom());
        auto height = static_cast<unsigned int>(screen_size.y*ed::GetCurrentZoom());

        auto layout_center = ed::ScreenToCanvas(CurrentEditorPos + (CurrentEditorSize / 2));//GetCanvasMousePosition();
        double k = 30;
        double energy_threshold = 1e-2;
        int iters_count = 300;

        nodesoup::adj_list_t g;
        std::unordered_map<void*, nodesoup::vertex_id_t> nodeId_vertexId;
        std::unordered_map<nodesoup::vertex_id_t, ed::NodeId> srg_nodes;
        auto name_to_vertex_id = [&g, &nodeId_vertexId, &srg_nodes](ed::NodeId node) -> nodesoup::vertex_id_t {

            auto name = node.AsPointer();
            nodesoup::vertex_id_t v_id;
            auto it = nodeId_vertexId.find(name);
            if (it != nodeId_vertexId.end()) {
                return (*it).second;
            }

            v_id = g.size();
            nodeId_vertexId.insert({name, v_id });
            srg_nodes.emplace(v_id, node);
            g.resize(v_id + 1);
            return v_id;
        };
        for (const auto& pattern : graph_editor_.Patterns) {
            // add vertex if new
            nodesoup::vertex_id_t v_id = name_to_vertex_id(pattern->DfgNode->ID);

        }

        for (const auto& edge : graph_editor_.DfgConnections) {
            auto start_pin = graph_editor_.FindPin(edge->StartPinID);
            nodesoup::vertex_id_t v_id = name_to_vertex_id(start_pin->ParentNode->ID);

            auto end_pin = graph_editor_.FindPin(edge->EndPinID);
            nodesoup::vertex_id_t adj_id = name_to_vertex_id(end_pin->ParentNode->ID);

            // add edge if new
            if (find(g[v_id].begin(), g[v_id].end(), adj_id) == g[v_id].end()) {
                g[v_id].push_back(adj_id);
                g[adj_id].push_back(v_id);
            }
        }

        std::vector<nodesoup::Point2D> positions = nodesoup::fruchterman_reingold(g, width, height, iters_count, k,
                                                                                  nullptr);


        for (nodesoup::vertex_id_t v_id = 0; v_id < g.size(); v_id++) {
            positions[v_id].x += layout_center.x;
            positions[v_id].y += layout_center.y;
        }
        for (nodesoup::vertex_id_t v_id = 0; v_id < g.size(); v_id++) {
            nodesoup::Point2D v_pos = positions[v_id];
            auto node = srg_nodes.at(v_id);

            ed::SetNodePosition(node, ImVec2(v_pos.x, v_pos.y));
        }
    }

    void DataflowFile::LayoutDFGNodesFromSinks() {
        std::vector<std::vector<editor::DFGNode::Ptr> > node_table;

        auto& nodes = graph_editor_.Patterns;
        auto node_count = nodes.size();

        std::vector<editor::DFGNode::Ptr> unsorted_nodes;
        unsorted_nodes.reserve(node_count);

        // sort out sinks and not output connected components for the first two columns
        // 0: sink (no output), 1: not connected output, rest unsorted
        node_table.resize(2);
        for (std::size_t i = 0; i < node_count; ++i) {
            auto& node = nodes[i]->DfgNode;
            node->UpdateInputWeight();
            if(node->Outputs.empty()) {
                node_table[0].emplace_back(node);
            } else {

                bool outputs_connected = true;

                for (const auto& input_pin : node->Outputs) {
                    outputs_connected = outputs_connected || input_pin->TraactPort->IsConnected();
                }

                if(!outputs_connected){
                    node_table[1].emplace_back(node);
                } else {
                    unsorted_nodes.emplace_back(node);
                }
            }
        }

        auto sort_by_input =  [](const editor::DFGNode::Ptr& a, const editor::DFGNode::Ptr& b) -> bool
        {
            return a->InputWeight > b->InputWeight;
        };

        sort(node_table[0].begin(), node_table[0].end(), sort_by_input);
        sort(node_table[1].begin(), node_table[1].end(), sort_by_input);

        std::size_t current_column = 2;
        auto is_prev_pattern = [&node_table, current_column](const pattern::instance::PatternInstance* a) -> bool
        {
            for (const auto& column : node_table) {
                auto result = std::find_if(column.cbegin(), column.cend(), [&a](const editor::DFGNode::Ptr b) -> bool
                {
                    return a == b->Pattern.get();
                });
                if(result != column.cend())
                    return true;
            }

            return false;
        };

        while(!unsorted_nodes.empty()){
            std::vector<editor::DFGNode::Ptr> next_column;
            std::vector<editor::DFGNode::Ptr> current_nodes;
            current_nodes.reserve(node_count);
            for (editor::DFGNode::Ptr& node : unsorted_nodes) {
                current_nodes.emplace_back(node);
            }

            for (editor::DFGNode::Ptr& node : current_nodes) {

                auto is_connected_to_only_prev = [&is_prev_pattern](const editor::DFGNode::Ptr& current_node){
                    for (const auto& output_pin : current_node->Outputs) {
                        if(!output_pin->TraactPort->IsConnected())
                            continue;
                        auto connected_to_ports = output_pin->TraactPort->connectedToPtr();

                        for(auto &connected_port : connected_to_ports){
                            auto source_pattern = connected_port->pattern_instance;
                            if(!is_prev_pattern(source_pattern)){
                                return false;
                            }
                        }
                    }

//                    for (const auto& input_pin : current_node->Inputs) {
//                        if(!input_pin->TraactPort->IsConnected())
//                            continue;
//                        auto connected_to_ports = input_pin->TraactPort->connectedToPtr();
//
//                        for(auto &connected_port : connected_to_ports){
//                            auto source_pattern = connected_port->pattern_instance;
//                            if(!is_prev_pattern(source_pattern)){
//                                return false;
//                            }
//                        }
//                    }

                    return true;
                };

                bool only_connected_to_prev = is_connected_to_only_prev(node);

                if(only_connected_to_prev) {
                    next_column.push_back(node);
                    auto remove_it = std::remove(unsorted_nodes.begin(),unsorted_nodes.end(), node);
                    unsorted_nodes.erase(remove_it, unsorted_nodes.end());
                }
            }



            sort(next_column.begin(), next_column.end(), sort_by_input);
            node_table.emplace_back(next_column);
            current_column++;
        }

        ImVec2 node_position(0,0);
        float padding = 10;
        float node_distance = 50;
        //for (auto& column : node_table) {
        for(auto it = node_table.rbegin(); it != node_table.rend(); ++it){
            float max_width = 0;
            for (auto cell : *it) {
                ed::SetNodePosition(cell->ID, node_position);
                auto node_size = ed::GetNodeSize(cell->ID);
                max_width = std::max(max_width, node_size.x);
                node_position.y += node_size.y + padding;
            }
            node_position.x += max_width + node_distance;
            node_position.y = 0;
        }
    }

    bool DataflowFile::CanUndo() const {
        return !undo_buffer_.empty();
    }

    bool DataflowFile::CanRedo() const {
        return !redo_buffer_.empty();
    }

    void DataflowFile::SaveState() {
        nlohmann::json json_graph;
        ns::to_json(json_graph, graph_editor_);
        undo_buffer_.push(json_graph);
        dirty = true;
    }

    void DataflowFile::Undo() {
        nlohmann::json json_graph;
        ns::to_json(json_graph, graph_editor_);
        redo_buffer_.push(json_graph);

        nlohmann::json state = undo_buffer_.top();
        graph_editor_.Graph = std::make_shared<DefaultInstanceGraph>();
        ns::from_json(state, graph_editor_);
        BuildNodes();
        undo_buffer_.pop();


    }

    void DataflowFile::Redo() {
        SaveState();

        nlohmann::json state = redo_buffer_.top();
        graph_editor_.Graph = std::make_shared<DefaultInstanceGraph>();
        ns::from_json(state, graph_editor_);
        BuildNodes();
        redo_buffer_.pop();
    }

    void DataflowFile::LayoutSRGNode(editor::EditorPattern::Ptr pattern) {
        auto screen_size = ed::GetScreenSize()/4;
        auto width = static_cast<unsigned int>(screen_size.x*ed::GetCurrentZoom());
        auto height = static_cast<unsigned int>(screen_size.y*ed::GetCurrentZoom());

        auto layout_center = GetCanvasMousePosition();
        double k = 15;
        double energy_threshold = 1e-2;
        int iters_count = 300;

        nodesoup::adj_list_t g;
        std::unordered_map<void*, nodesoup::vertex_id_t> nodeId_vertexId;
        std::unordered_map<nodesoup::vertex_id_t, editor::SRGNode::Ptr> srg_nodes;
        auto name_to_vertex_id = [&g, &nodeId_vertexId, &srg_nodes](editor::SRGNode::Ptr node) -> nodesoup::vertex_id_t {

            auto name = node->ID.AsPointer();
            nodesoup::vertex_id_t v_id;
            auto it = nodeId_vertexId.find(name);
            if (it != nodeId_vertexId.end()) {
                return (*it).second;
            }

            v_id = g.size();
            nodeId_vertexId.insert({name, v_id });
            srg_nodes.emplace(v_id, node);
            g.resize(v_id + 1);
            return v_id;
        };

        for (const auto& edge : pattern->SrgConnections) {
            if(!edge->IsVisible())
                continue;

            // add vertex if new
            nodesoup::vertex_id_t v_id = name_to_vertex_id(edge->SourceNode);

            // add adjacent vertex if new
            nodesoup::vertex_id_t adj_id = name_to_vertex_id(edge->TargetNode);

            // add edge if new
            if (find(g[v_id].begin(), g[v_id].end(), adj_id) == g[v_id].end()) {
                g[v_id].push_back(adj_id);
                g[adj_id].push_back(v_id);
            }

        }

        std::vector<nodesoup::Point2D> positions = nodesoup::fruchterman_reingold(g, width, height, iters_count, k,
                                                                                  nullptr);


        for (nodesoup::vertex_id_t v_id = 0; v_id < g.size(); v_id++) {
            positions[v_id].x += layout_center.x;
            positions[v_id].y += layout_center.y;
        }
        for (nodesoup::vertex_id_t v_id = 0; v_id < g.size(); v_id++) {
            nodesoup::Point2D v_pos = positions[v_id];
            auto node = srg_nodes.at(v_id);

            node->Position = ImVec2(v_pos.x, v_pos.y);
            ed::SetNodePosition(node->ID, node->Position);

        }
    }


}


