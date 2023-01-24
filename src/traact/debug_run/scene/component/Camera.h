/** Copyright (C) 2022  Frieder Pankratz <frieder.pankratz@gmail.com> **/

#ifndef TRAACT_GUI_SRC_TRAACT_DEBUG_RUN_SCENE_COMPONENT_CAMERA_H_
#define TRAACT_GUI_SRC_TRAACT_DEBUG_RUN_SCENE_COMPONENT_CAMERA_H_

#include <glm/glm.hpp>
#include <glm/ext.hpp>
#include "traact/debug_run/scene/Component.h"

namespace traact::gui::scene::component {

class Camera : public Component {
 public:
    Camera(const std::shared_ptr<Object> &object,std::string name);
 private:
    virtual void draw() override;
 public:
    virtual void drawGui() override;
 public:
    void updateMovement();
    void moveForward();
    void moveBackward();
    void moveShiftLeft();
    void moveShiftRight();
    void moveUp();
    void moveDown();
    void mouseRotate(float dx, float dy);
    glm::mat4 getViewMatrix() const;
    glm::mat4 getProjectionMatrix() const;
    glm::mat4 getVPMatrix() const;

    const float* getViewMatrixPtr() const;
    const float* getProjectionMatrixPtr() const;

 private:
    glm::mat4 projection_;
    glm::mat4 view_;

    float move_speed_{1.0f};
    float rotate_speed_{0.1f};
    float fov_{60.0f};

    glm::vec3 camera_pos_{0.0f, 0.0f,  3.0f};
    glm::vec3 camera_front_{0.0f, 0.0f, -1.0f};
    glm::vec3 camera_up_{0.0f, 1.0f,  0.0f};

    float camera_yaw_{-90.0f};
    float camera_pitch_{0};

    float delta_time_{0};
    float last_frame_{0};

    float getMoveSpeed() const;
    void updateProjection();
};

} // traact

#endif //TRAACT_GUI_SRC_TRAACT_DEBUG_RUN_SCENE_COMPONENT_CAMERA_H_
