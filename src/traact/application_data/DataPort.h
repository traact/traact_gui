/** Copyright (C) 2022  Frieder Pankratz <frieder.pankratz@gmail.com> **/

#ifndef TRAACT_GUI_SRC_TRAACT_APPLICATION_DATA_DATAPORT_H_
#define TRAACT_GUI_SRC_TRAACT_APPLICATION_DATA_DATAPORT_H_

#include <traact/traact.h>

namespace traact::application_data {

class ApplicationData;

class DataPort {
 public:
    DataPort(ApplicationData *application_data, pattern::instance::PortInstance::ConstPtr const &port);
    virtual ~DataPort() = default;
    virtual bool processTimePoint(traact::buffer::ComponentBuffer &data) = 0;
    bool isInitialized() const;
 protected:
    ApplicationData* application_data_;
    bool initialized_{false};
    int port_index_;

};

} // traact

#endif //TRAACT_GUI_SRC_TRAACT_APPLICATION_DATA_DATAPORT_H_
