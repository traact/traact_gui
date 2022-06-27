/** Copyright (C) 2022  Frieder Pankratz <frieder.pankratz@gmail.com> **/

#include "EditorUtils.h"
#include <atomic>
namespace traact::gui::editor::utils {
    int GetNextId()
    {
        static std::atomic<int> nextId(0);
        return nextId++;
    }

//    void ShowLabel(const char* label, ImColor color)
//    {
//        using namespace ImGui;
//        ImGui::SetCursorPosY(GetCursorPosY() - GetTextLineHeight());
//        auto size = ImGui::CalcTextSize(label);
//
//        auto padding = ImGui::GetStyle().FramePadding;
//        auto spacing = ImGui::GetStyle().ItemSpacing;
//
//        ImGui::SetCursorPos(GetCursorPos() + ImVec2(spacing.x, -spacing.y));
//
//        auto rectMin = ImGui::GetCursorScreenPos() - padding;
//        auto rectMax = ImGui::GetCursorScreenPos() + size + padding;
//
//        auto drawList = GetWindowDrawList();
//        drawList->AddRectFilled(rectMin, rectMax, color, size.y * 0.15f);
//        ImGui::TextUnformatted(label);
//    };
}
