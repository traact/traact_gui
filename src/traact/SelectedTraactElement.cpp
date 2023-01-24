/** Copyright (C) 2022  Frieder Pankratz <frieder.pankratz@gmail.com> **/

#include "SelectedTraactElement.h"
#include "DataflowFile.h"

namespace traact::gui {

void SelectedTraactElement::clearSelection() {
    selected = {};
}
bool SelectedTraactElement::isCurrentDataflow(const std::shared_ptr<DataflowFile> dataflow) {
    return current_dataflow == dataflow;
}

template<>
void SelectedTraactElement::setSelected(const std::shared_ptr<DataflowFile> &selected_element) {
    if (isSelected(selected_element)) {
        return;
    }
    SPDLOG_TRACE("set selected dataflow to {0}", selected_element->getName());
    selected = selected_element;
    current_dataflow = selected_element;
}


} // traact