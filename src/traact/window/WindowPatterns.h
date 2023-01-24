/** Copyright (C) 2022  Frieder Pankratz <frieder.pankratz@gmail.com> **/

#ifndef TRAACT_GUI_SRC_TRAACT_WINDOW_WINDOWPATTERNS_H_
#define TRAACT_GUI_SRC_TRAACT_WINDOW_WINDOWPATTERNS_H_

#include "traact/app/Window.h"
#include "traact/state/project/pattern/PatternInfo.h"
#include <array>
#include <re2/re2.h>
namespace traact::gui::window {

class WindowPatterns : public Window{
 public:
    WindowPatterns(state::ApplicationState &state);
    void init() override;
    void render() override;
 private:
    struct TreeItem {
        TreeItem(std::string t_name);
        std::string name;
        std::vector<TreeItem> children;
        std::vector<size_t> patterns;
    };

    std::vector<state::project::PatternInfo> all_patterns_;
    std::vector<bool> show_pattern_;

    std::vector<TreeItem > tree_items_;
    TreeItem unassigned_items_;

    std::array<char,64> filter_;


    void drawPattern(size_t pattern_index);
    void filterPatterns();
    bool showPattern(int pattern_index, re2::RE2& regex_filter);
    void updatePatternData();
    void drawTreeItem(const TreeItem &tree_item);
    bool addPatternToNode(size_t pattern_index, TreeItem &tree_item);
};

} // traact

#endif //TRAACT_GUI_SRC_TRAACT_WINDOW_WINDOWPATTERNS_H_
