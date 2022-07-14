/** Copyright (C) 2022  Frieder Pankratz <frieder.pankratz@gmail.com> **/

#ifndef TRAACT_GUI_SRC_TRAACT_GUI_STATE_OPENFILES_H_
#define TRAACT_GUI_SRC_TRAACT_GUI_STATE_OPENFILES_H_

#include <vector>

#include <traact/util/CircularBuffer.h>
namespace traact::gui::state {

struct OpenFiles{
    util::CircularBuffer<std::string, 5> recent_files_;
    std::vector<std::string> open_files_;
};

} // traact

#endif //TRAACT_GUI_SRC_TRAACT_GUI_STATE_OPENFILES_H_
