/** Copyright (C) 2022  Frieder Pankratz <frieder.pankratz@gmail.com> **/

#include <Magnum/Platform/GLContext.h>
#include <Magnum/GL/Renderer.h>

#include "imgui.h"
#define IMGUI_DEFINE_MATH_OPERATORS
#include "imgui_internal.h"

#include "Window.h"

#include "feature/DrawTriangle.h"
#include <Magnum/Platform/GLContext.h>
#include <Magnum/ImGuiIntegration/Context.hpp>
Magnum::Platform::GLContext* kGlobalContex{nullptr};

namespace traact::gui::magnum {
void Window::init() {
    using namespace Magnum;
    using namespace Magnum::Math::Literals;

//
//        Vector2{windowSize()}/dpiScaling(),
//        windowSize(), framebufferSize());

    //contex_ = new Magnum::Platform::GLContext();

    if(!kGlobalContex){
        kGlobalContex = new Magnum::Platform::GLContext();
    }

    Magnum::Platform::GLContext::makeCurrent(kGlobalContex);


    framebuffer_.setViewPort(render_width_,render_height_);
    framebuffer_.init();

    /* Configure camera */
    camera_object_ = new Object3D{&scene_};
    camera_object_->translate(Magnum::Vector3::zAxis(5.0f));
    camera_ = new Magnum::SceneGraph::Camera3D{*camera_object_};
    camera_->setAspectRatioPolicy(Magnum::SceneGraph::AspectRatioPolicy::Extend)
        .setProjectionMatrix(Magnum::Matrix4::perspectiveProjection(35.0_degf, 4.0f/3.0f, 0.001f, 100.0f))
        .setViewport(Magnum::Vector2i(render_width_, render_height_));

    auto& test_child = scene_.addChild<DrawTriangle>();
    test_child.setParent(camera_object_);
    test_child.translate(Magnum::Vector3::zAxis(-1.0f));



    using namespace Magnum;
    using namespace Magnum::Math::Literals;


//
//    struct TriangleVertex {
//        Vector2 position;
//        Color3 color;
//    };
//    const TriangleVertex vertices[]{
//        {{-0.5f, -0.5f}, 0xff0000_rgbf},    /* Left vertex, red color */
//        {{ 0.5f, -0.5f}, 0x00ff00_rgbf},    /* Right vertex, green color */
//        {{ 0.0f,  0.5f}, 0x0000ff_rgbf}     /* Top vertex, blue color */
//    };
//    _mesh.setCount(Containers::arraySize(vertices))
//        .addVertexBuffer(GL::Buffer{vertices}, 0,
//                         Shaders::VertexColor2D::Position{},
//                         Shaders::VertexColor2D::Color3{});


}
void Window::update(buffer::ComponentBuffer &buffer,
                                       std::vector<RenderCommand> &additional_commands) {

}
void Window::draw() {


    ImGui::Begin("MagnumScene", nullptr);

    renderMagnum();


    glBindTexture(GL_TEXTURE_2D, framebuffer_.getRenderTexture());
    ImVec2 avail_size = ImGui::GetContentRegionAvail();
    ImVec2 p = ImGui::GetCursorScreenPos();
    ImGui::Image(reinterpret_cast<void *>( static_cast<intptr_t>( framebuffer_.getRenderTexture() )), avail_size, ImVec2(0,1), ImVec2(1,0));



    ImGui::End();

}
void Window::renderMagnum() {
    using namespace Magnum;

    framebuffer_.bind();
    /* Reset state. Only needed if you want to draw something else with
       different state after. */
    GL::Renderer::enable(GL::Renderer::Feature::DepthTest);
    GL::Renderer::enable(GL::Renderer::Feature::FaceCulling);
    GL::Renderer::disable(GL::Renderer::Feature::ScissorTest);
    GL::Renderer::disable(GL::Renderer::Feature::Blending);


    camera_->draw(drawables_);

    drawFoo();

    /* Set appropriate states. If you only draw ImGui, it is sufficient to
       just enable blending and scissor test in the constructor. */
    GL::Renderer::enable(GL::Renderer::Feature::Blending);
    GL::Renderer::enable(GL::Renderer::Feature::ScissorTest);
    GL::Renderer::disable(GL::Renderer::Feature::FaceCulling);
    GL::Renderer::disable(GL::Renderer::Feature::DepthTest);
    framebuffer_.resetBind();
}
void Window::update() {

}
void Window::drawFoo() {


    //_shader.draw(_mesh);
}

} // traact