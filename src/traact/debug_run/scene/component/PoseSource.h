/** Copyright (C) 2022  Frieder Pankratz <frieder.pankratz@gmail.com> **/

#ifndef TRAACT_GUI_SRC_TRAACT_DEBUG_RUN_SCENE_COMPONENT_PoseSource_H_
#define TRAACT_GUI_SRC_TRAACT_DEBUG_RUN_SCENE_COMPONENT_PoseSource_H_

#include <open3d/Open3D.h>
#include <cuda_gl_interop.h>
#include "traact/debug_run/scene/Component.h"
#include <opencv2/core/opengl.hpp>
#include "traact/opengl_shader.h"
#include <cuda_runtime.h>
#include "traact/application_data/application_data.h"
namespace traact::gui::scene::component {

class PoseSource : public Component{
 public:
    PoseSource(const std::shared_ptr<Object> &object,std::string name);
    void setSource(application_data::PoseSourcePtr source);

    void update() override;

 private:
    application_data::PoseSourcePtr pose_source_;

};

} // traact

#endif //TRAACT_GUI_SRC_TRAACT_DEBUG_RUN_SCENE_COMPONENT_PoseSource_H_
