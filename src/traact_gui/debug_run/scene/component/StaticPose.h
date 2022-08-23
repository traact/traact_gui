/** Copyright (C) 2022  Frieder Pankratz <frieder.pankratz@gmail.com> **/

#ifndef TRAACT_GUI_SRC_TRAACT_GUI_DEBUG_RUN_SCENE_COMPONENT_STATICPOSE_H_
#define TRAACT_GUI_SRC_TRAACT_GUI_DEBUG_RUN_SCENE_COMPONENT_STATICPOSE_H_

#include "traact_gui/debug_run/scene/Component.h"
#include <traact/traact.h>

namespace traact::gui::scene::component  {

class StaticPose : public Component{
 public:
    StaticPose(const std::shared_ptr<Object> &object, const std::string &name);
    void setPattern(const std::shared_ptr<traact::pattern::instance::PatternInstance> &pattern_instance);

    void drawGui() override;
 private:

    std::shared_ptr<traact::pattern::instance::PatternInstance> pattern_instance_;

    void saveCalibration();
};

} // traact

#endif //TRAACT_GUI_SRC_TRAACT_GUI_DEBUG_RUN_SCENE_COMPONENT_STATICPOSE_H_
