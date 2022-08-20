/** Copyright (C) 2022  Frieder Pankratz <frieder.pankratz@gmail.com> **/

#ifndef TRAACT_GUI_SRC_TRAACT_GUI_SELECTEDTRAACTELEMENT_H_
#define TRAACT_GUI_SRC_TRAACT_GUI_SELECTEDTRAACTELEMENT_H_

#include <variant>
#include <traact/pattern/instance/GraphInstance.h>
#include "editor/DFGElements.h"

namespace traact::gui {

struct DataflowFile;

template<class... Ts> struct overloaded : Ts... { using Ts::operator()...; };
template<class... Ts> overloaded(Ts...) -> overloaded<Ts...>;

using TraactElement = std::variant<std::shared_ptr<DataflowFile>, std::shared_ptr<traact::pattern::instance::PatternInstance> >;

struct SelectedTraactElement {
    SelectedTraactElement() = default;
    template<typename T> SelectedTraactElement(const T& value){
        setSelected(value);
    }
    ~SelectedTraactElement() = default;

    TraactElement selected;
    std::shared_ptr<DataflowFile> current_dataflow;

    template<typename T>
    void setSelected(const T &selected_element) {
        if (isSelected(selected_element)) {
            return;
        }
        SPDLOG_TRACE("set selected element to {0}", selected_element->getName());
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

    bool isCurrentDataflow(const std::shared_ptr<DataflowFile> dataflow);

};

template<>
void SelectedTraactElement::setSelected(const std::shared_ptr<DataflowFile> &selected_element);



} // traact

#endif //TRAACT_GUI_SRC_TRAACT_GUI_SELECTEDTRAACTELEMENT_H_
