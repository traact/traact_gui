/** Copyright (C) 2022  Frieder Pankratz <frieder.pankratz@gmail.com> **/

#include <Magnum/Math/Color.h>
#include "DrawTriangle.h"
#include <spdlog/spdlog.h>
namespace traact::gui::magnum {

DrawTriangle::DrawTriangle(Magnum::SceneGraph::Object<Magnum::SceneGraph::MatrixTransformation3D> *parent)
    : Object(parent), Magnum::SceneGraph::Drawable3D(*this) {

    using namespace Magnum;
    using namespace Magnum::Math::Literals;

    struct TriangleVertex {
        Vector2 position;
        Color3 color;
    };
    const TriangleVertex vertices[]{
        {{-0.5f, -0.5f}, 0xff0000_rgbf},    /* Left vertex, red color */
        {{ 0.5f, -0.5f}, 0x00ff00_rgbf},    /* Right vertex, green color */
        {{ 0.0f,  0.5f}, 0x0000ff_rgbf}     /* Top vertex, blue color */
    };

    _mesh.setCount(Containers::arraySize(vertices))
        .addVertexBuffer(GL::Buffer{vertices}, 0,
                         Shaders::VertexColor2D::Position{},
                         Shaders::VertexColor2D::Color3{});
}

void DrawTriangle::draw(const Magnum::MatrixTypeFor<3, Magnum::Float> &transformation_matrix,
                        Magnum::SceneGraph::Camera<3, Magnum::Float> &camera) {



    SPDLOG_INFO("raw triangle");


    _shader.draw(_mesh);
}

} // traact