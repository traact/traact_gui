/** Copyright (C) 2022  Frieder Pankratz <frieder.pankratz@gmail.com> **/

#ifndef TRAACT_GUI_SRC_TRAACT_GUI_SELECTEDTRAACTELEMENT_H_
#define TRAACT_GUI_SRC_TRAACT_GUI_SELECTEDTRAACTELEMENT_H_

#include <variant>
#include <traact/pattern/instance/GraphInstance.h>
#include "editor/DFGElements.h"

namespace traact::gui {

struct DataflowFile;

struct SelectedTraactElement {
    std::variant<std::shared_ptr<DataflowFile>, std::shared_ptr<traact::pattern::instance::PatternInstance> >selected;

    template<typename T>
    void setSelected(const T &selected_element) {
        if (isSelected(selected_element)) {
            return;
        }
        SPDLOG_INFO("set seltected element {0}", selected_element->getName());
        selected = selected_element;
    }

    void clearSelection();

    template<typename T>
    bool isSelected(const T &graph_instance) const {
        if (std::holds_alternative<T>(selected)) {
            return graph_instance == std::get<T>(selected);
        } else {
            return false;
        }
    }

};

} // traact

#endif //TRAACT_GUI_SRC_TRAACT_GUI_SELECTEDTRAACTELEMENT_H_
