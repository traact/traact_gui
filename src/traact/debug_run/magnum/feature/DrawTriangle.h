/** Copyright (C) 2022  Frieder Pankratz <frieder.pankratz@gmail.com> **/

#ifndef TRAACT_GUI_SRC_TRAACT_DEBUG_RUN_MAGNUM_FEATURE_DRAWTRIANGLE_H_
#define TRAACT_GUI_SRC_TRAACT_DEBUG_RUN_MAGNUM_FEATURE_DRAWTRIANGLE_H_

#include "traact/debug_run/magnum/magnum_definitions.h"
#include <Magnum/SceneGraph/Drawable.h>
#include <Magnum/GL/Mesh.h>
#include <Magnum/Shaders/VertexColor.h>

namespace traact::gui::magnum {

class DrawTriangle : public Object3D, Magnum::SceneGraph::Drawable3D {
 public:
    DrawTriangle(Object<Magnum::SceneGraph::MatrixTransformation3D> *parent);
 private:
    Magnum::GL::Mesh _mesh;
    Magnum::Shaders::VertexColor2D _shader;

    virtual void draw(const Magnum::MatrixTypeFor<3, Magnum::Float> &transformation_matrix,
                      Magnum::SceneGraph::Camera<3, Magnum::Float> &camera) override;

};


} // traact

#endif //TRAACT_GUI_SRC_TRAACT_DEBUG_RUN_MAGNUM_FEATURE_DRAWTRIANGLE_H_
