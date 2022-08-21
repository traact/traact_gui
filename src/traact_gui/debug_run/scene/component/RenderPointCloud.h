/** Copyright (C) 2022  Frieder Pankratz <frieder.pankratz@gmail.com> **/

#ifndef TRAACT_GUI_SRC_TRAACT_GUI_DEBUG_RUN_SCENE_COMPONENT_RENDERPOINTCLOUD_H_
#define TRAACT_GUI_SRC_TRAACT_GUI_DEBUG_RUN_SCENE_COMPONENT_RENDERPOINTCLOUD_H_

#include <open3d/Open3D.h>
#include <cuda_gl_interop.h>
#include "traact_gui/debug_run/scene/Component.h"
#include <opencv2/core/opengl.hpp>
#include "traact_gui/opengl_shader.h"
#include <cuda_runtime.h>
#include "traact_gui/application_data/application_data.h"

namespace traact::gui::scene::component {

class RenderPointCloud : public Component{
 public:
    RenderPointCloud(const std::shared_ptr<Object> &object,std::string name);
    void setVertexSource(application_data::TextureSourcePtr source);
    void setColorSource(application_data::TextureSourcePtr source);

    void update() override;

    void draw() override;
    virtual void drawGui() override;
 private:
    application_data::TextureSourcePtr cloud_vertex_;
    application_data::TextureSourcePtr cloud_color_;
    GLuint texture_vertex_;
    GLuint texture_color_;
    bool init_texture_{false};
    size_t num_points_{0};
    std::atomic_bool has_data_{false};
    Shader shader_;
    bool render_initialized_{false};
    int port_index_vertex_;
    int port_index_color_;
    Eigen::Vector4f add_color_{0,0,0,1};
    float point_size_{0.01f};

    cudaStream_t stream_;

    cudaGraphicsResource *cuda_resource_vertex_;
    cudaGraphicsResource *cuda_resource_color_;

};

} // traact

#endif //TRAACT_GUI_SRC_TRAACT_GUI_DEBUG_RUN_SCENE_COMPONENT_RENDERPOINTCLOUD_H_
