/** Copyright (C) 2022  Frieder Pankratz <frieder.pankratz@gmail.com> **/

#ifndef TRAACT_GUI_SRC_TRAACT_GUI_DEBUG_RUN_SCENE_COMPONENT_RENDERPOINTCLOUD_H_
#define TRAACT_GUI_SRC_TRAACT_GUI_DEBUG_RUN_SCENE_COMPONENT_RENDERPOINTCLOUD_H_

#include <open3d/Open3D.h>
#include <cuda_gl_interop.h>
#include "traact_gui/debug_run/scene/Component.h"
#include <opencv2/core/opengl.hpp>
#include "traact_gui/opengl_shader.h"
#include <cuda_runtime.h>

namespace traact::gui::scene::component {

class RenderPointCloud : public Component{
 public:
    RenderPointCloud(const std::shared_ptr<Object> &object,std::string name);
    void setVertexPort(int port_index);
    void setColorPort(int port_index);

    void update(buffer::ComponentBuffer &buffer, std::vector<RenderCommand> &additional_commands) override;

    void draw() override;
 private:
    GLuint texture_vertex_;
    GLuint texture_color_;
    bool init_texture_{false};
    size_t num_points_{0};
    std::atomic_bool has_data_{false};
    Shader shader_;
    bool render_initialized_{false};
    int port_index_vertex_;
    int port_index_color_;

    cudaStream_t stream_;

    cudaGraphicsResource *cuda_resource_vertex_;
    cudaGraphicsResource *cuda_resource_color_;

    void upload(buffer::ComponentBuffer &data);

};

} // traact

#endif //TRAACT_GUI_SRC_TRAACT_GUI_DEBUG_RUN_SCENE_COMPONENT_RENDERPOINTCLOUD_H_
