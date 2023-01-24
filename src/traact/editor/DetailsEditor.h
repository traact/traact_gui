/** Copyright (C) 2022  Frieder Pankratz <frieder.pankratz@gmail.com> **/

#ifndef TRAACT_GUI_SRC_TRAACT_EDITOR_DETAILSEDITOR_H_
#define TRAACT_GUI_SRC_TRAACT_EDITOR_DETAILSEDITOR_H_

#include <traact/pattern/instance/GraphInstance.h>
#include "DFGElements.h"
#include "traact/DataflowFile.h"
#include "traact/SelectedTraactElement.h"

namespace traact::gui {

struct DetailsEditor {
    void operator()(std::shared_ptr<DataflowFile>& dataflow_file) const ;
    void operator()(std::shared_ptr<traact::pattern::instance::PatternInstance>& pattern_instance) const;

    std::function<void(const TraactElement&)> onChange;
};

} // traact

#endif //TRAACT_GUI_SRC_TRAACT_EDITOR_DETAILSEDITOR_H_
