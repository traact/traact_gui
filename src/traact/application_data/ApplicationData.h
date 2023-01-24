/** Copyright (C) 2022  Frieder Pankratz <frieder.pankratz@gmail.com> **/

#ifndef TRAACT_GUI_SRC_TRAACT_APPLICATION_DATA_APPLICATIONDATA_H_
#define TRAACT_GUI_SRC_TRAACT_APPLICATION_DATA_APPLICATIONDATA_H_

#include <cuda_runtime.h>
#include <traact/traact.h>
#include "DataPort.h"

namespace traact::application_data {

class ApplicationData {
 public:
    ApplicationData() = default;
    ~ApplicationData() = default;

    void init();
    void destroy();

    bool processTimePoint(traact::buffer::ComponentBuffer &buffer);

    template<class DataBufferType>
    std::shared_ptr<DataBufferType> addDataPort(pattern::instance::PortInstance::ConstPtr const&port){
        auto new_buffer = std::make_shared<DataBufferType>(this, port);
        data_buffers_.push_back(new_buffer);
        return new_buffer;
    }

    cudaStream_t getCudaStream();
 private:

    cudaStream_t stream_;
    std::vector<std::shared_ptr<DataPort>> data_buffers_;

    bool initialized_{false};

};

} // traact

#endif //TRAACT_GUI_SRC_TRAACT_APPLICATION_DATA_APPLICATIONDATA_H_
