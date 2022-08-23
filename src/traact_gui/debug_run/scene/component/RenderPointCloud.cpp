/** Copyright (C) 2022  Frieder Pankratz <frieder.pankratz@gmail.com> **/

#include <traact/point_cloud.h>
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
uniform vec4 add_color;
layout(rgba32f, binding=0) readonly uniform image2D pointCloudTexture;
layout(rgba8ui, binding=1) readonly uniform uimage2D pointCloudColorTexture;

void main()
{

    ivec2 pointCloudSize = imageSize(pointCloudTexture);
    ivec2 currentDepthPixelCoordinates = ivec2(gl_VertexID % pointCloudSize.x, gl_VertexID / pointCloudSize.x);
    vec4 vertexPosition = imageLoad(pointCloudTexture, currentDepthPixelCoordinates);

    // opencv to opengl
    vertexPosition.y *= -1;
    vertexPosition.z *= -1;
    gl_Position = mvp * vertexPosition;


    uvec3 color = imageLoad(pointCloudColorTexture, currentDepthPixelCoordinates).xyz;
    vertexColor.r = color.r /255.0f;
    vertexColor.g = color.g /255.0f;
    vertexColor.b = color.b /255.0f;
    vertexColor.a = 1.0f;

    vertexColor.r = clamp(vertexColor.r + add_color.r, 0,1);
    vertexColor.g = clamp(vertexColor.g + add_color.g, 0,1);
    vertexColor.b = clamp(vertexColor.b + add_color.b, 0,1);
    //vertexColor.a = clamp(vertexColor.a + add_color.a, 0,1);


    // Pass along the 'invalid pixel' flag as the alpha channel
    //
    if (vertexPosition.z == 0.0f)
    {
        vertexColor.a = 0.0f;
    }
}
)";

// clang-format on

namespace traact::gui::scene::component {
RenderPointCloud::RenderPointCloud(const std::shared_ptr<Object> &object,std::string name) : Component(object, std::move(name)) {

}

void RenderPointCloud::update() {

}

void RenderPointCloud::draw() {



    if (!render_initialized_) {
        shader_.init(kVertexShader, kFragmentShader);

        render_initialized_ = true;
    }

    if (!cloud_vertex_) {
        SPDLOG_ERROR("source for point cloud vertex not set");
        return;
    }
    if (!cloud_color_) {
        SPDLOG_ERROR("source for point cloud color not set");
        return;
    }

    if (!cloud_vertex_->isInitialized() || !cloud_color_->isInitialized()) {
        return;
    }

    glMatrixMode(GL_MODELVIEW);

    checkOpenGLErrors();
    checkOpenGLErrors(glEnable(GL_DEPTH_TEST));
    checkOpenGLErrors(glEnable(GL_BLEND));
    checkOpenGLErrors(glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA));
    checkOpenGLErrors(glPointSize(static_cast<GLfloat>(point_size_)));
    shader_.use();


    checkOpenGLErrors();

    cloud_vertex_->bind(0);

    shader_.setUniform("pointCloudTexture", 0);

    cloud_color_->bind(1);
    shader_.setUniform("pointCloudColorTexture", 1);

    auto mvp = camera_->getVPMatrix() * transform_->getWorldPose();
    shader_.setUniform("mvp", glm::value_ptr(mvp));

    shader_.setUniform("add_color", add_color_.x(), add_color_.y(), add_color_.z(), add_color_.w());


    checkOpenGLErrors(glDrawArrays(GL_POINTS, 0, cloud_vertex_->getHeight()*cloud_vertex_->getWidth()));

    checkOpenGLErrors(glBindVertexArray(0));
    glUseProgram(0);


}
void RenderPointCloud::setVertexSource(application_data::TextureSourcePtr source) {
    cloud_vertex_ = source;
}
void RenderPointCloud::setColorSource(application_data::TextureSourcePtr source) {
    cloud_color_ = source;

}
void RenderPointCloud::drawGui() {
    ImGui::ColorEdit4("Additional Color", add_color_.data());
    ImGui::SliderFloat("Point Size", &point_size_, 0.0f,0.1);
}
} // traact