/** Copyright (C) 2022  Frieder Pankratz <frieder.pankratz@gmail.com> **/

#ifndef TRAACT_GUI_SRC_TRAACT_WINDOW_WINDOWLOG_H_
#define TRAACT_GUI_SRC_TRAACT_WINDOW_WINDOWLOG_H_

#include "traact/app/Window.h"
#include "external/ImTerm/terminal.hpp"
#include "external/ImTerm/terminal_commands.hpp"

namespace traact::gui::window {

class WindowLog : public Window{

 public:
    WindowLog(state::ApplicationState &state);
    void render() override;
    bool renderStop() override;
 private:
    custom_command_struct cmd_struct_; // terminal commands can interact with this structure
    ImTerm::terminal<terminal_commands> terminal_log_;
    bool showing_term_{true};

};

} // traact

#endif //TRAACT_GUI_SRC_TRAACT_WINDOW_WINDOWLOG_H_
