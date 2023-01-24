/** Copyright (C) 2022  Frieder Pankratz <frieder.pankratz@gmail.com> **/

#ifndef TRAACT_GUI_SRC_TRAACT_DEBUG_RUN_MAGNUM_MAGNUM_DEFINITIONS_H_
#define TRAACT_GUI_SRC_TRAACT_DEBUG_RUN_MAGNUM_MAGNUM_DEFINITIONS_H_

#include <Magnum/SceneGraph/Scene.h>
#include <Magnum/SceneGraph/MatrixTransformation3D.h>

namespace traact::gui::magnum {

using Object3D = Magnum::SceneGraph::Object<Magnum::SceneGraph::MatrixTransformation3D>;
using Scene3D = Magnum::SceneGraph::Scene<Magnum::SceneGraph::MatrixTransformation3D>;

}
#endif //TRAACT_GUI_SRC_TRAACT_DEBUG_RUN_MAGNUM_MAGNUM_DEFINITIONS_H_
