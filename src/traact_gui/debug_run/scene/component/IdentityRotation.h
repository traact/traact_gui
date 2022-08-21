/** Copyright (C) 2022  Frieder Pankratz <frieder.pankratz@gmail.com> **/

#ifndef TRAACT_GUI_SRC_TRAACT_GUI_DEBUG_RUN_SCENE_COMPONENT_IDENTITYROTATION_H_
#define TRAACT_GUI_SRC_TRAACT_GUI_DEBUG_RUN_SCENE_COMPONENT_IDENTITYROTATION_H_

#include "traact_gui/debug_run/scene/Component.h"

namespace traact::gui::scene::component  {

class IdentityRotation : public Component{
 public:
    IdentityRotation(const std::shared_ptr<Object> &object, const std::string &name);
    virtual void update() override;

};

} // traact

#endif //TRAACT_GUI_SRC_TRAACT_GUI_DEBUG_RUN_SCENE_COMPONENT_IDENTITYROTATION_H_
