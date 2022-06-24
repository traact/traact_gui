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
#include "JsonEditorSerialization.h"
#include <traact/serialization/JsonGraphInstance.h>

void ns::to_json(json &jobj, const traact::gui::editor::PatternGraphEditor &obj) {
    to_json(jobj, *obj.Graph);

    json editor_patterns;
    for (const auto &pattern_instance : obj.Patterns) {
        json editor_pattern;
        json dfg_node;
        to_json(dfg_node, *pattern_instance->DfgNode);

        editor_pattern["dfg_node"] = dfg_node;

        json srg_nodes;
        for (const auto &node : pattern_instance->SrgNodes) {
            json pattern_node;
            to_json(pattern_node, *node);
            srg_nodes[node->CoordinateSystem->name] = pattern_node;
        }

        editor_pattern["srg_nodes"] = srg_nodes;

        editor_patterns[pattern_instance->Pattern->instance_id] = editor_pattern;
    }
    jobj["editor_patterns"] = editor_patterns;




    json srg_merge_nodes;
    for (const auto &merge_node : obj.SrgMergedNodes) {
        json tmp;
        to_json(tmp, *merge_node);
        srg_merge_nodes.emplace_back(tmp);
    }
    jobj["srg_merge_nodes"] = srg_merge_nodes;

    json srg_merge_edges;
    for (const auto &merged_edge : obj.SrgMergedEdges) {
        json tmp;
        to_json(tmp, *merged_edge);
        srg_merge_edges.emplace_back(tmp);
    }
    jobj["srg_merge_edges"] = srg_merge_edges;

}

void ns::from_json(const json &jobj, traact::gui::editor::PatternGraphEditor &obj) {
    from_json(jobj, *obj.Graph);

    obj.CreateNodes();
    obj.UpdateSRGGraph();

}

void ns::to_json(nlohmann::json &jobj, const traact::gui::editor::DFGNode &obj) {
    to_json(jobj["GUIPos"], obj.Position);
}

void ns::from_json(const nlohmann::json &jobj, traact::gui::editor::DFGNode &obj) {
    from_json(jobj["GUIPos"], obj.Position);
}

void ns::to_json(nlohmann::json &jobj, const traact::gui::editor::SRGNode &obj) {
    to_json(jobj["GUIPos"], obj.Position);
}

void ns::from_json(const nlohmann::json &jobj, traact::gui::editor::SRGNode &obj) {
    from_json(jobj["GUIPos"], obj.Position);
}

void ns::to_json(nlohmann::json &jobj, const traact::gui::editor::SRGMergedNode &obj) {
    jobj["name"] = obj.Name;
    to_json(jobj["GUIPos"], obj.Position);
    json node_children;
    for(const auto& child : obj.Children){
        node_children.emplace_back(child->GetName());
    }
    jobj["children"] = node_children;
}

void ns::from_json(const nlohmann::json &jobj, traact::gui::editor::SRGMergedNode &obj) {
    obj.Name = jobj["name"].get<std::string>();
    from_json(jobj["GUIPos"], obj.Position);
}

void ns::to_json(nlohmann::json &jobj, const traact::gui::editor::SRGMergedEdge &obj) {

}

void ns::from_json(const nlohmann::json &jobj, traact::gui::editor::SRGMergedEdge &obj) {

}

void ns::to_json(nlohmann::json &jobj, const ImVec2 &obj) {
    jobj["x"] = obj.x;
    jobj["y"] = obj.y;
}

void ns::from_json(const nlohmann::json &jobj, ImVec2 &obj) {
    jobj["x"].get_to(obj.x);
    jobj["y"].get_to(obj.y);
}

