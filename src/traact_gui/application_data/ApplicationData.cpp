/** Copyright (C) 2022  Frieder Pankratz <frieder.pankratz@gmail.com> **/

#include "ApplicationData.h"
#include "traact_gui/debug_run/traact_opengl.h"

namespace traact {

void application_data::ApplicationData::init() {
    checkCudaErrors(cudaStreamCreate(&stream_));
}
void application_data::ApplicationData::destroy() {
    checkCudaErrors(cudaStreamDestroy(stream_));
}
bool application_data::ApplicationData::processTimePoint(buffer::ComponentBuffer &buffer) {

    for (auto& tmp : data_buffers_) {
        tmp->processTimePoint(buffer);
    }
    return false;
}
} // traact