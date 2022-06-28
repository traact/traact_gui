/** Copyright (C) 2022  Frieder Pankratz <frieder.pankratz@gmail.com> **/

#ifndef TRAACT_GUI_SRC_TRAACT_GUI_EDITOR_DETAILSEDITOR_H_
#define TRAACT_GUI_SRC_TRAACT_GUI_EDITOR_DETAILSEDITOR_H_

#include <traact/pattern/instance/GraphInstance.h>
#include "DFGElements.h"
#include "traact_gui/DataflowFile.h"

namespace traact::gui {

struct DetailsEditor {
    void operator()(std::shared_ptr<DataflowFile>& dataflow_file) const ;
    void operator()(std::shared_ptr<traact::pattern::instance::PatternInstance>& pattern_instance) const;

    std::function<void(void)> onChange;
};

} // traact

#endif //TRAACT_GUI_SRC_TRAACT_GUI_EDITOR_DETAILSEDITOR_H_
