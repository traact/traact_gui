/** Copyright (C) 2022  Frieder Pankratz <frieder.pankratz@gmail.com> **/

#ifndef TRAACT_GUI_SRC_TRAACT_GUI_APPLICATION_DATA_APPLICATION_DATA_H_
#define TRAACT_GUI_SRC_TRAACT_GUI_APPLICATION_DATA_APPLICATION_DATA_H_

#include <traact/spatial.h>
#include <traact/vision.h>
#include <traact/pointCloud.h>
#include "ApplicationData.h"
#include "source/RawSource.h"

namespace traact::application_data {
    using PoseSource = application_data::source::RawSource<spatial::Pose6DHeader>;
    using PoseSourcePtr = std::shared_ptr<PoseSource>;
}

#endif //TRAACT_GUI_SRC_TRAACT_GUI_APPLICATION_DATA_APPLICATION_DATA_H_
