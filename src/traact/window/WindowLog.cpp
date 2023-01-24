/** Copyright (C) 2022  Frieder Pankratz <frieder.pankratz@gmail.com> **/

#include "WindowLog.h"

namespace traact::gui::window{
WindowLog::WindowLog(state::ApplicationState &state) : Window("Logs", state),
                                                       terminal_log_(cmd_struct_, "Logs"){
    spdlog::default_logger()->sinks().push_back(terminal_log_.get_terminal_helper());
    terminal_log_.log_level(ImTerm::message::severity::info);
    terminal_log_.execute("configure_terminal colors set-theme \"Dark Cherry\"");
    //terminal_log_.execute("configure_terminal completion to-top");
    terminal_log_.set_max_log_len(5000);
}
void WindowLog::render() {


    if (showing_term_) {
        showing_term_ = terminal_log_.show();
    }
}
bool WindowLog::renderStop() {
    render();
    return false;
}
} // traact