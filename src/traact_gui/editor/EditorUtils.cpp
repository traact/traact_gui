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
