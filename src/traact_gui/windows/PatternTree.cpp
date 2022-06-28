/** Copyright (C) 2022  Frieder Pankratz <frieder.pankratz@gmail.com> **/

#include "PatternTree.h"
#include <imgui.h>

traact::gui::PatternTree::PatternTree(traact::gui::TraactGuiApp *traactApp) : traact_app_(traactApp) {}

void traact::gui::PatternTree::Draw() {

    ImGui::PushStyleVar(ImGuiStyleVar_WindowMinSize, ImVec2(250, 250));
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
    ImGui::PopStyleVar();

}
