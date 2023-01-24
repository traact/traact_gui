/** Copyright (C) 2022  Frieder Pankratz <frieder.pankratz@gmail.com> **/

#ifndef TRAACTMULTI_JSONEDITORSERIALIZATION_H
#define TRAACTMULTI_JSONEDITORSERIALIZATION_H


#include <nlohmann/json.hpp>
#include "PatternGraphEditor.h"


namespace ns {

    using nlohmann::json;

    void to_json(json &jobj, const ImVec2 &obj) ;

    void from_json(const json &jobj, ImVec2 &obj);

    void to_json(json &jobj, const traact::gui::editor::DFGNode &obj) ;

    void from_json(const json &jobj, traact::gui::editor::DFGNode &obj);


    void to_json(json &jobj, const traact::gui::editor::SRGNode &obj) ;

    void from_json(const json &jobj, traact::gui::editor::SRGNode &obj);

    void to_json(json &jobj, const traact::gui::editor::SRGMergedNode &obj) ;

    void from_json(const json &jobj, traact::gui::editor::SRGMergedNode &obj);

    void to_json(json &jobj, const traact::gui::editor::SRGMergedEdge &obj) ;

    void from_json(const json &jobj, traact::gui::editor::SRGMergedEdge &obj);



    void to_json(json &jobj, const traact::gui::editor::PatternGraphEditor &obj) ;

    void from_json(const json &jobj, traact::gui::editor::PatternGraphEditor &obj);
} // namespace ns

#endif //TRAACTMULTI_JSONEDITORSERIALIZATION_H
