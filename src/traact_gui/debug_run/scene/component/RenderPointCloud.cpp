/** Copyright (C) 2022  Frieder Pankratz <frieder.pankratz@gmail.com> **/

#include <traact/pointCloud.h>
#include "RenderPointCloud.h"
#include <glm/glm.hpp>
#include "traact_gui/debug_run/traact_opengl.h"
#include "Camera.h"





// clang-format off

constexpr char const kFragmentShader[] =
    R"(
#version 430

in vec4 vertexColor;
out vec4 fragmentColor;

void main()
{
    if (vertexColor.a == 0.0f)
    {
        discard;
    }

    fragmentColor = vertexColor;
}
)";

constexpr char const kVertexShader[] =
    R"(
#version 430

//layout(location = 0) in vec4 inPosition;
out vec4 vertexColor;
uniform mat4 mvp;
layout(rgba32f, binding=0) readonly uniform image2D pointCloudTexture;
layout(rgba8ui, binding=1) readonly uniform uimage2D pointCloudColorTexture;

void main()
{
    //ivec2 pointCloudSize = ivec2(100,100);
    ivec2 pointCloudSize = imageSize(pointCloudTexture);
    ivec2 currentDepthPixelCoordinates = ivec2(gl_VertexID % pointCloudSize.x, gl_VertexID / pointCloudSize.x);
    vec4 vertexPosition = imageLoad(pointCloudTexture, currentDepthPixelCoordinates);

    //gl_Position = mvp * vec4(currentDepthPixelCoordinates.x/10.0f, currentDepthPixelCoordinates.y/10.0f, 0,1);
    // opencv to opengl
    vertexPosition.y *= -1;
    vertexPosition.z *= -1;
    gl_Position = mvp * vertexPosition;


    uvec3 color = imageLoad(pointCloudColorTexture, currentDepthPixelCoordinates).xyz;
    vertexColor.r = color.r /255.0f;
    vertexColor.g = color.g /255.0f;
    vertexColor.b = color.b /255.0f;


    // Pass along the 'invalid pixel' flag as the alpha channel
    //
//    if (vertexPosition.z == 0.0f)
//    {
//        vertexColor.a = 0.0f;
//    }
}
)";

// clang-format on

namespace traact::gui::scene::component {
RenderPointCloud::RenderPointCloud(const std::shared_ptr<Object> &object,std::string name) : Component(object, std::move(name)) {

}

void RenderPointCloud::update(buffer::ComponentBuffer &buffer, std::vector<RenderCommand> &additional_commands) {
    if (buffer.isInputValid(port_index_vertex_)) {
        additional_commands.emplace_back([this, &buffer]() {
            upload(buffer);
        });
    }
}
void RenderPointCloud::upload(buffer::ComponentBuffer &data) {


    const auto &buffer_vertex = data.getInput<vision::GpuImageHeader>(port_index_vertex_).value();
    const auto &image_header_vertex = data.getInputHeader<vision::GpuImageHeader>(port_index_vertex_);

    const auto &buffer_color = data.getInput<vision::GpuImageHeader>(port_index_color_).value();
    const auto &image_header_color = data.getInputHeader<vision::GpuImageHeader>(port_index_color_);

    if (!init_texture_) {
        checkOpenGLErrors(glGenTextures(1, &texture_vertex_));
        checkOpenGLErrors(glBindTexture(GL_TEXTURE_2D, texture_vertex_));
        checkOpenGLErrors(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR));
        checkOpenGLErrors(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR));
        checkOpenGLErrors(glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, image_header_vertex.width, image_header_vertex.height, 0,
                                       GL_RGBA, GL_UNSIGNED_BYTE, 0));
        checkCudaErrors(cudaGraphicsGLRegisterImage(&cuda_resource_vertex_, texture_vertex_, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsWriteDiscard));
        checkOpenGLErrors(glBindTexture(GL_TEXTURE_2D, 0));

        num_points_ = image_header_vertex.width * image_header_vertex.height;

        checkOpenGLErrors(glGenTextures(1, &texture_color_));
        checkOpenGLErrors(glBindTexture(GL_TEXTURE_2D, texture_color_));
        checkOpenGLErrors(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR));
        checkOpenGLErrors(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR));
        checkOpenGLErrors(glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8UI, image_header_color.width, image_header_color.height, 0,
                                       GL_RGBA_INTEGER, GL_UNSIGNED_BYTE, 0));
        checkCudaErrors(cudaGraphicsGLRegisterImage(&cuda_resource_color_, texture_color_, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsWriteDiscard));
        checkOpenGLErrors(glBindTexture(GL_TEXTURE_2D, 0));

        checkCudaErrors(cudaStreamCreate(&stream_));

        init_texture_ = true;
    }


    checkCudaErrors(cudaGraphicsMapResources(1, &cuda_resource_vertex_, stream_));
    checkCudaErrors(cudaGraphicsMapResources(1, &cuda_resource_color_,stream_));

    {
        cudaArray *texture_ptr;
        checkCudaErrors(cudaGraphicsSubResourceGetMappedArray(&texture_ptr, cuda_resource_vertex_, 0, 0));
        checkCudaErrors(cudaMemcpy2DToArrayAsync(texture_ptr, 0, 0, buffer_vertex.cudaPtr(), buffer_vertex.step, buffer_vertex.step, buffer_vertex.rows, cudaMemcpyDeviceToDevice, stream_));
    }
    {
        cudaArray *texture_ptr;
        checkCudaErrors(cudaGraphicsSubResourceGetMappedArray(&texture_ptr, cuda_resource_color_, 0, 0));
        checkCudaErrors(cudaMemcpy2DToArrayAsync(texture_ptr, 0, 0, buffer_color.cudaPtr(), buffer_color.step, buffer_color.step, buffer_color.rows, cudaMemcpyDeviceToDevice,stream_));
    }

    checkCudaErrors(cudaGraphicsUnmapResources(1, &cuda_resource_vertex_,stream_));
    checkCudaErrors(cudaGraphicsUnmapResources(1, &cuda_resource_color_,stream_));

    checkCudaErrors(cudaStreamSynchronize(stream_));

    has_data_.store(true);



}
void RenderPointCloud::draw() {
    if (!render_initialized_) {
        shader_.init(kVertexShader, kFragmentShader);

        render_initialized_ = true;
    }

    if (!has_data_.load()) {
        return;
    }

    glMatrixMode(GL_MODELVIEW);

    checkOpenGLErrors();
    checkOpenGLErrors(glEnable(GL_DEPTH_TEST));
    checkOpenGLErrors(glEnable(GL_BLEND));
    checkOpenGLErrors(glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA));
    checkOpenGLErrors(glPointSize(static_cast<GLfloat>(0.01)));
    shader_.use();


    //point_cloud_.bind(cv::ogl::Buffer::ARRAY_BUFFER);
    checkOpenGLErrors();

    checkOpenGLErrors(glActiveTexture(GL_TEXTURE0));
    checkOpenGLErrors(glBindTexture(GL_TEXTURE_2D, texture_vertex_));
    checkOpenGLErrors(glBindImageTexture(0,
                                         texture_vertex_,
                                         0,
                                         GL_FALSE,
                                         0,
                                         GL_READ_ONLY,
                                         GL_RGBA32F));

    shader_.setUniform("pointCloudTexture", 0);

    checkOpenGLErrors(glActiveTexture(GL_TEXTURE0+1));
    checkOpenGLErrors(glBindTexture(GL_TEXTURE_2D, texture_color_));
    checkOpenGLErrors(glBindImageTexture(1,
                                         texture_color_,
                                         0,
                                         GL_FALSE,
                                         0,
                                         GL_READ_ONLY,
                                         GL_RGBA8UI));
    shader_.setUniform("pointCloudColorTexture", 1);

    auto mvp = camera_->getVPMatrix() * transform_->getWorldPose();
    shader_.setUniform("mvp", glm::value_ptr(mvp));


    checkOpenGLErrors(glDrawArrays(GL_POINTS, 0, num_points_));

    checkOpenGLErrors(glBindVertexArray(0));
    glUseProgram(0);


}
void RenderPointCloud::setVertexPort(int port_index) {
    port_index_vertex_ = port_index;
}
void RenderPointCloud::setColorPort(int port_index) {
    port_index_color_ = port_index;

}
} // traact