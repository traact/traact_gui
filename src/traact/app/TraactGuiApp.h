/** Copyright (C) 2022  Frieder Pankratz <frieder.pankratz@gmail.com> **/

#ifndef TRAACTMULTI_TRAACTGUIAPP_H
#define TRAACTMULTI_TRAACTGUIAPP_H

#include <vector>
#include <string>
#include <map>
#include <traact/pattern/instance/GraphInstance.h>
#include "traact/DataflowFile.h"
#include "util/fileutil.h"
#include <traact/util/CircularBuffer.h>
#include "traact/SelectedTraactElement.h"

#include "traact/debug_run/DebugRun.h"

#include "traact/state/ApplicationState.h"
#include "Window.h"


namespace traact::gui {

    class TraactGuiApp {
    public:
        TraactGuiApp();

        ~TraactGuiApp();

        void addWindow(Window::SharedPtr window);

        void init();
        void render();
        bool renderStop();



     private:
        std::vector<Window::SharedPtr> windows_;
        std::vector<bool> render_windows_;

        void menuBar();

        void onComponentPropertyChange();
    };
}



#endif //TRAACTMULTI_TRAACTGUIAPP_H
