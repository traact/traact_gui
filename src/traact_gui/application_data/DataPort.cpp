/** Copyright (C) 2022  Frieder Pankratz <frieder.pankratz@gmail.com> **/

#include "DataPort.h"

namespace traact::application_data {
DataPort::DataPort(ApplicationData *application_data, pattern::instance::PortInstance::ConstPtr const &port)
    : application_data_(application_data), port_index_(port->getPortIndex()) {}
bool DataPort::isInitialized() const {
    return initialized_;
}
} // traact