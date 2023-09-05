/** Copyright (C) 2022  Frieder Pankratz <frieder.pankratz@gmail.com> **/

#include "WindowPatterns.h"
#include "external/ImTerm/misc.hpp"
#include <string_view>
namespace traact::gui::window {
WindowPatterns::WindowPatterns(state::ApplicationState &state) : Window("Patterns", state), unassigned_items_("unable to sort") {}
void WindowPatterns::init() {

    auto& source_tag = tree_items_.emplace_back(pattern::tags::kSource);
    source_tag.children.emplace_back(pattern::tags::kApplication);
    auto& function_tag = tree_items_.emplace_back(pattern::tags::kFunction);
    auto& sink_tag = tree_items_.emplace_back(pattern::tags::kSink);
    sink_tag.children.emplace_back(pattern::tags::kApplication);

    updatePatternData();


}
void WindowPatterns::render() {
    static constexpr ImGuiTreeNodeFlags base_flags = ImGuiTreeNodeFlags_OpenOnArrow | ImGuiTreeNodeFlags_SpanAvailWidth;

    ImGui::Begin("Patterns");

    if(ImGui::InputTextWithHint("##Pattern_Filter", "filter...", filter_.data(), filter_.size())){
        filterPatterns();


    };


    for (const auto& item : tree_items_) {
        drawTreeItem(item);
    }
    drawTreeItem(unassigned_items_);

    ImGui::End();
}
void WindowPatterns::drawTreeItem(const WindowPatterns::TreeItem &tree_item) {
    if(tree_item.patterns.empty() && tree_item.children.empty()){
        return;
    }
    ImGui::SetNextItemOpen(true, ImGuiCond_Once);
    if (ImGui::TreeNode(tree_item.name.c_str())) {
        for(const auto& child : tree_item.children){
            drawTreeItem(child);
        }
        for(const auto& pattern_index : tree_item.patterns){
            drawPattern(pattern_index);
        }
        ImGui::TreePop();
    }

}

void WindowPatterns::drawPattern(size_t pattern_index) {
    if(show_pattern_[pattern_index]){
        const auto& pattern = all_patterns_[pattern_index];
        ImGui::Selectable(pattern.display_name.c_str());
        if(ImGui::IsItemHovered()){
            ImGui::BeginTooltip();
            ImGui::Text("Pattern ID: %s", pattern.pattern_id.c_str());
            if(pattern.description.empty()){
                ImGui::Text("No description");
            } else {
                ImGui::TextWrapped("Description: \n%s", pattern.description.c_str());
            }
            std::stringstream tags_stream;
            for(const auto& tag : pattern.tags){
                tags_stream << tag << " ";
            }
            ImGui::TextWrapped("Tags: \n%s", tags_stream.str().c_str());

            ImGui::EndTooltip();
        }

        if (ImGui::BeginDragDropSource(ImGuiDragDropFlags_None)) {
            ImGui::SetDragDropPayload("NEW_PATTERN_DRAGDROP", pattern.pattern_id.c_str(), pattern.pattern_id.length() + 1);
            ImGui::EndDragDropSource();
        }
    }
}
void WindowPatterns::filterPatterns() {
    std::string_view filter_view(filter_.data(), misc::strnlen(filter_.data(), filter_.size()));
    std::fill(show_pattern_.begin(), show_pattern_.end(), true);
    if(!filter_view.empty()){

        std::string filter_string(filter_view);
        std::stringstream filter_stream(filter_string);
        std::string segment;

        while (std::getline(filter_stream, segment, ' ')) {
            std::string regex_string = fmt::format("(?i){0}.*", segment);
            SPDLOG_DEBUG("using pattern filter regex: {0}", regex_string);
            re2::RE2 regex_filter(regex_string);

            if(regex_filter.ok()) {
                for (int i = 0; i < all_patterns_.size(); ++i) {
                    if(show_pattern_[i]){
                        show_pattern_[i] = showPattern(i, regex_filter);
                    }
                }
            } else {
                SPDLOG_ERROR("regex for filtering of pattern invalid: {0}", regex_string);
            }
        }
    }
}
bool WindowPatterns::showPattern(int pattern_index, re2::RE2& regex_filter) {
    const auto& pattern = all_patterns_[pattern_index];


    if(RE2::PartialMatch(pattern.display_name.c_str(), regex_filter)){
        return true;
    }

    if(RE2::PartialMatch(pattern.pattern_id.c_str(), regex_filter)){
        return true;
    }
    for(const auto& tag : pattern.tags){
        if(RE2::PartialMatch(tag.c_str(), regex_filter)){
            return true;
        }
    }
    return false;
}
void WindowPatterns::updatePatternData() {
    all_patterns_ = state_.getAvailablePatterns();

    show_pattern_.resize(all_patterns_.size(), true);
    std::fill(show_pattern_.begin(), show_pattern_.end(),true);

    for (size_t pattern_index = 0; pattern_index < all_patterns_.size(); ++pattern_index) {
        bool was_assigned{false};
        for (auto& root_item : tree_items_) {
            was_assigned |= addPatternToNode(pattern_index, root_item);
        }

        if(!was_assigned){
            unassigned_items_.patterns.push_back(pattern_index);
        }
    }
}
bool WindowPatterns::addPatternToNode(size_t pattern_index, WindowPatterns::TreeItem &tree_item) {
    auto& pattern = all_patterns_[pattern_index];
    if(pattern.containsTag(tree_item.name)){
        bool assigned_to_child{false};
        for(auto& child : tree_item.children){
            assigned_to_child |= addPatternToNode(pattern_index, child);
        }
        if(!assigned_to_child){
            tree_item.patterns.push_back(pattern_index);
        }
        return true;
    } else {
        return false;
    }

}


WindowPatterns::TreeItem::TreeItem(std::string t_name) : name(t_name){

}
} // traact