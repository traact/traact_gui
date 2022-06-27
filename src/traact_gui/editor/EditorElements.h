/** Copyright (C) 2022  Frieder Pankratz <frieder.pankratz@gmail.com> **/

#ifndef TRAACTMULTI_EDITORELEMENTS_H
#define TRAACTMULTI_EDITORELEMENTS_H

#include "DFGElements.h"
#include "SRGElements.h"


namespace traact::gui::editor {

    struct EditorPattern {
        typedef typename std::shared_ptr<EditorPattern> Ptr;
        explicit EditorPattern(pattern::instance::PatternInstance::Ptr patternInstance);
        pattern::instance::PatternInstance::Ptr Pattern;
        DFGNode::Ptr DfgNode;

        std::vector<SRGNode::Ptr> SrgNodes;
        std::vector<SRGEdge::Ptr> SrgConnections;
    };



}


#endif //TRAACTMULTI_EDITORELEMENTS_H
