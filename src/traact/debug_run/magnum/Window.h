/** Copyright (C) 2022  Frieder Pankratz <frieder.pankratz@gmail.com> **/

#ifndef TRAACT_GUI_SRC_TRAACT_DEBUG_RUN_MAGNUM_WINDOW_H_
#define TRAACT_GUI_SRC_TRAACT_DEBUG_RUN_MAGNUM_WINDOW_H_

//#include <Magnum/GL/Mesh.h>
//#include <Magnum/Shaders/VertexColor.h>

#include <Magnum/Math/Color.h>
#include <Magnum/SceneGraph/Camera.h>
#include <Magnum/SceneGraph/Drawable.h>
#include <spdlog/spdlog.h>

#include <traact/buffer/ComponentBuffer.h>
#include "traact/opengl/Framebuffer.h"
#include "traact/debug_run/DebugRenderComponent.h"

#include "magnum_definitions.h"


namespace traact::gui::magnum {


class Window {
 public:

    void init();

    void update();
    void draw();

    void update(buffer::ComponentBuffer &buffer, std::vector<RenderCommand> &additional_commands);

 private:
//    Magnum::GL::Mesh _mesh;
//    Magnum::Shaders::VertexColor2D _shader;
    Scene3D scene_;
    Object3D* camera_object_;
    Magnum::SceneGraph::Camera3D* camera_;
    Magnum::SceneGraph::DrawableGroup3D drawables_;
    opengl::Framebuffer framebuffer_;
    int render_width_{1280};
    int render_height_{720};


    void renderMagnum();
    void drawFoo();
};

} // traact

#endif //TRAACT_GUI_SRC_TRAACT_DEBUG_RUN_MAGNUM_WINDOW_H_
