/** Copyright (C) 2022  Frieder Pankratz <frieder.pankratz@gmail.com> **/

#ifndef TRAACT_GUI_SRC_TRAACT_DEBUG_RUN_SCENE_TRANSFORM_H_
#define TRAACT_GUI_SRC_TRAACT_DEBUG_RUN_SCENE_TRANSFORM_H_

#include <memory>
#include <glm/glm.hpp>
#include <glm/ext.hpp>

namespace traact::gui::scene {

class Transform {
 public:
    using SharedPtr = std::shared_ptr<Transform>;
    Transform() = default;
    ~Transform();

    [[nodiscard]] glm::mat4 getLocalPose() const;
    [[nodiscard]] float* getLocalPosePtr() ;
    [[nodiscard]] glm::mat4 getWorldPose() const;
    void setLocalPose(glm::mat4 pose);

    void setParent(SharedPtr parent);
    SharedPtr getParent();
 private:
    glm::mat4 pose_{1};
    SharedPtr parent_{nullptr};
};

} // traact

#endif //TRAACT_GUI_SRC_TRAACT_DEBUG_RUN_SCENE_TRANSFORM_H_
