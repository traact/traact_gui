/** Copyright (C) 2022  Frieder Pankratz <frieder.pankratz@gmail.com> **/

#ifndef TRAACTMULTI_TRAACTGUIAPP_H
#define TRAACTMULTI_TRAACTGUIAPP_H

#include <vector>
#include <string>
#include <map>
#include <traact/pattern/instance/GraphInstance.h>
#include <traact_gui/DataflowFile.h>
#include <util/fileutil.h>
#include <traact/util/CircularBuffer.h>
#include "SelectedTraactElement.h"
#include "editor/DetailsEditor.h"
#include "debug_run/DebugRun.h"

namespace traact::gui {

    class TraactGuiApp {
    public:
        TraactGuiApp(std::string config_file);

        ~TraactGuiApp();

        void onFrame();

        void newFile();
        void newFile(const std::string& dataflow_json);
        void openFile(fs::path file);
        const std::vector<std::string> & openFiles();
        void closeFile(std::string file);
        void closeAll();

        bool onFrameStop();
     private:
        std::string config_file_;
        util::CircularBuffer<std::string, 5> recent_files_;
        std::vector<std::string> open_files_;

        std::vector<pattern::Pattern::Ptr> available_patterns_;

        std::vector<std::shared_ptr<DataflowFile>> dataflow_files_;
        std::shared_ptr<DataflowFile> current_dataflow_{nullptr};
        std::shared_ptr<DataflowFile> pending_dataflow_{nullptr};
        SelectedTraactElement selected_traact_element_;
        DetailsEditor details_editor_;
        DebugRun debug_run_;

        void menuBar();

        void drawLeftPanel();
        void drawDataflowPanel();

        void drawDataflowFilesPanel();
        void drawDetailsPanel();

        void saveConfig();
        void loadConfig();

        void onComponentPropertyChange();
    };
}



#endif //TRAACTMULTI_TRAACTGUIAPP_H
