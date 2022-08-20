/** Copyright (C) 2022  Frieder Pankratz <frieder.pankratz@gmail.com> **/

#ifndef TRAACT_GUI_SRC_TRAACT_GUI_DEBUG_RUN_SCENE_COMPONENT_RENDERCOORDINATESYSTEM_H_
#define TRAACT_GUI_SRC_TRAACT_GUI_DEBUG_RUN_SCENE_COMPONENT_RENDERCOORDINATESYSTEM_H_

#include "traact_gui/debug_run/scene/Component.h"

namespace traact::gui::scene::component {

class RenderCoordinateSystem : public Component{

 public:
    RenderCoordinateSystem(const std::shared_ptr<Object> &object,std::string name);
    virtual void draw() override;
    virtual void drawGui() override;

    float scale_{1.0f};
};

} // traact

#endif //TRAACT_GUI_SRC_TRAACT_GUI_DEBUG_RUN_SCENE_COMPONENT_RENDERCOORDINATESYSTEM_H_
