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

#include "PatternTree.h"
#include <imgui.h>

traact::gui::PatternTree::PatternTree(traact::gui::TraactGuiApp *traactApp) : traact_app_(traactApp) {}

void traact::gui::PatternTree::Draw() {

    ImGui::Begin("Pattern");

    //const auto& graphs = traact_app_->GetGraphInstances();
    //const auto& pattern = traact_app_->GetAvailablePatterns();

    if(ImGui::TreeNode("Loaded Graph")){

//        for(const auto& name_graph : graphs){
//            const auto& graph = name_graph.second;
//            if(ImGui::TreeNode(graph->name.c_str())){
//                for(const auto& tmp : graph->getAll()){
//                    ImGui::Text(tmp->getPatternName().c_str());
//                }
//
//                ImGui::TreePop();
//            }
//        }

        ImGui::TreePop();
    }
    if(ImGui::TreeNode("All Patterns")){
//        for (const auto& tmp : pattern) {
//            ImGui::Button(tmp->name.c_str());
//        }
        ImGui::TreePop();
    }

    ImGui::End();

}
