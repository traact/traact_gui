/** Copyright (C) 2022  Frieder Pankratz <frieder.pankratz@gmail.com> **/

#ifndef TRAACT_GUI_SRC_TRAACT_GUI_APPLICATIONSTATE_H_
#define TRAACT_GUI_SRC_TRAACT_GUI_APPLICATIONSTATE_H_

#include <nlohmann/json.hpp>
#include <traact/util/CircularBuffer.h>
#include "traact_gui/DataflowFile.h"
#include "traact_gui/SelectedTraactElement.h"

namespace traact::gui::state {

struct ApplicationState {
    explicit ApplicationState(const std::string &config_file);
    std::string config_file;
    util::CircularBuffer<std::string, 5> recent_files;
    std::vector<std::shared_ptr<DataflowFile>> open_files;

    std::vector<std::optional<std::string>> open_file_paths;

    std::vector<pattern::Pattern::Ptr> available_patterns;
    SelectedTraactElement selected_traact_element;

    nlohmann::json config;
    void update();
    void saveConfig();
    void loadConfig();
    void newFile();
    void newFile(const std::string& dataflow_json);
    void openFile(fs::path file);
    const std::vector<std::optional<std::string>> &openFiles();
    void closeFile(std::string file);
    void closeAll();
    bool selectionChanged();

    void dataflowDetailsChanged(const TraactElement &changed_element);

    bool selectionChangedTo(const std::shared_ptr<DataflowFile> &dataflow);
 private:
    std::vector<std::shared_ptr<DataflowFile>> close_queue;
    std::stack<std::shared_ptr<DataflowFile>> save_queue;
    std::shared_ptr<DataflowFile> last_selected_dataflow;
};

} // traact

#endif //TRAACT_GUI_SRC_TRAACT_GUI_APPLICATIONSTATE_H_
