/** Copyright (C) 2022  Frieder Pankratz <frieder.pankratz@gmail.com> **/

#include "ApplicationData.h"
#include "traact/opengl/traact_opengl.h"

namespace traact {

void application_data::ApplicationData::init() {
    checkCudaErrors(cudaStreamCreate(&stream_));
    initialized_ = true;
}
void application_data::ApplicationData::destroy() {
    checkCudaErrors(cudaStreamDestroy(stream_));
}
bool application_data::ApplicationData::processTimePoint(buffer::ComponentBuffer &buffer) {

    if(!initialized_){
        init();
    }

    for (auto& tmp : data_buffers_) {
        tmp->processTimePoint(buffer);
    }
    checkCudaErrors(cudaStreamSynchronize(stream_));
    return false;
}
cudaStream_t application_data::ApplicationData::getCudaStream() {
    return stream_;
}
} // traact