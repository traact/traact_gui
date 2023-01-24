/** Copyright (C) 2022  Frieder Pankratz <frieder.pankratz@gmail.com> **/

#ifndef TRAACT_GUI_SRC_TRAACT_DEBUG_RUN_SCENE_COMPONENT_H_
#define TRAACT_GUI_SRC_TRAACT_DEBUG_RUN_SCENE_COMPONENT_H_

#include "traact/debug_run/DebugRenderComponent.h"
#include "glm/glm.hpp"
#include "Transform.h"
#include "imgui.h"
#define IMGUI_DEFINE_MATH_OPERATORS
#include "imgui_internal.h"

namespace traact::gui::scene {

class Object;

namespace component {
    class Camera;
}


class Component {
 public:
    using SharedPtr = std::shared_ptr<Component>;
    explicit Component(std::shared_ptr<Object> object, std::string name);
    virtual void update();
    virtual void draw();
    virtual void drawGui();

    bool active{true};
    const char* getName() const;
    const std::shared_ptr<Object> &getObject() const;
    const std::shared_ptr<Transform> &getTransform() const;

 protected:
    std::shared_ptr<Object> object_;
    std::shared_ptr<Transform> transform_;
    std::shared_ptr<component::Camera> camera_;
    std::string name_;
};

} // traact

#endif //TRAACT_GUI_SRC_TRAACT_DEBUG_RUN_SCENE_COMPONENT_H_
