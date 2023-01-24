/** Copyright (C) 2022  Frieder Pankratz <frieder.pankratz@gmail.com> **/

#ifndef TRAACTMULTI_IMGUIUTILS_H
#define TRAACTMULTI_IMGUIUTILS_H

#include <imgui.h>
#define IMGUI_DEFINE_MATH_OPERATORS
#include <imgui_internal.h>
#include <external/imgui-node-editor/imgui_node_editor.h>

static bool Splitter(const char* splitter_id, bool split_vertically, float thickness, float* size1, float* size2, float min_size1, float min_size2, float splitter_long_axis_size = -1.0f)
{
    using namespace ImGui;
    ImGuiContext& g = *GImGui;
    ImGuiWindow* window = g.CurrentWindow;
    ImGuiID id = window->GetID(splitter_id);
    ImRect bb;
    bb.Min = window->DC.CursorPos + (split_vertically ? ImVec2(*size1, 0.0f) : ImVec2(0.0f, *size1));
    bb.Max = bb.Min + CalcItemSize(split_vertically ? ImVec2(thickness, splitter_long_axis_size) : ImVec2(splitter_long_axis_size, thickness), 0.0f, 0.0f);
    return SplitterBehavior(bb, id, split_vertically ? ImGuiAxis_X : ImGuiAxis_Y, size1, size2, min_size1, min_size2, 0.0f);
}

static ImVec2 GetCanvasMousePosition() {
    auto mouse_pos = ImGui::GetMousePos();
    return ax::NodeEditor::ScreenToCanvas(mouse_pos);
}

static void ShowLabel(const char* label, ImColor color, bool atCanvasMousePos = false)
{
    using namespace ImGui;
    ImGui::SetCursorPosY(GetCursorPosY() - GetTextLineHeight());
    auto size = ImGui::CalcTextSize(label);

    auto padding = ImGui::GetStyle().FramePadding;
    auto spacing = ImGui::GetStyle().ItemSpacing;

    if(atCanvasMousePos)
        ImGui::SetCursorPos(GetCanvasMousePosition() + ImVec2(spacing.x, -spacing.y));
    else
        ImGui::SetCursorPos(GetCursorPos() + ImVec2(spacing.x, -spacing.y));

    auto rectMin = ImGui::GetCursorScreenPos() - padding;
    auto rectMax = ImGui::GetCursorScreenPos() + size + padding;

    auto drawList = GetWindowDrawList();
    drawList->AddRectFilled(rectMin, rectMax, color, size.y * 0.15f);
    ImGui::TextUnformatted(label);
};



#endif //TRAACTMULTI_IMGUIUTILS_H
