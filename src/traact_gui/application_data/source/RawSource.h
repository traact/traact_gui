/** Copyright (C) 2022  Frieder Pankratz <frieder.pankratz@gmail.com> **/

#ifndef TRAACT_GUI_SRC_TRAACT_GUI_APPLICATION_DATA_SOURCE_RAWSOURCE_H_
#define TRAACT_GUI_SRC_TRAACT_GUI_APPLICATION_DATA_SOURCE_RAWSOURCE_H_

#include "traact_gui/application_data/DataPort.h"

namespace traact::application_data::source {

template<class THeader>
class RawSource : public DataPort {
 public:

    RawSource(ApplicationData *application_data, pattern::instance::PortInstance::ConstPtr const &port) : DataPort(
        application_data,
        port) {};

    bool processTimePoint(buffer::ComponentBuffer &data) override {
        if (data.isInputValid(port_index_)) {
            buffer_ = data.template getInput<THeader>(port_index_);
        }
        return true;
    }

    const typename THeader::NativeType &getBuffer() {
        return buffer_;
    }

 private:
    typename THeader::NativeType buffer_;

};

} // traact

#endif //TRAACT_GUI_SRC_TRAACT_GUI_APPLICATION_DATA_SOURCE_RAWSOURCE_H_
