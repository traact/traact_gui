/** Copyright (C) 2022  Frieder Pankratz <frieder.pankratz@gmail.com> **/

#include "Window.h"

namespace traact::gui {

Window::Window(std::string name, state::ApplicationState &state) : state_(state), name_(name) {}
void Window::init() {

}
void Window::render() {

}
bool Window::renderStop() {
    return false;
}
void Window::destroy() {

}
const std::string &Window::name() const{
    return name_;
}
}