/*  BSD 3-Clause License
 *
 *  Copyright (c) 2020, FriederPankratz <frieder.pankratz@gmail.com>
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions are met:
 *
 *  1. Redistributions of source code must retain the above copyright notice, this
 *     list of conditions and the following disclaimer.
 *
 *  2. Redistributions in binary form must reproduce the above copyright notice,
 *     this list of conditions and the following disclaimer in the documentation
 *     and/or other materials provided with the distribution.
 *
 *  3. Neither the name of the copyright holder nor the names of its
 *     contributors may be used to endorse or promote products derived from
 *     this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 *  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 *  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 *  DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 *  FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 *  DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 *  SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 *  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 *  OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 *  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
**/

#include "TraactGuiApp.h"

#include <traact/facade/DefaultFacade.h>
#include <traact/serialization/JsonGraphInstance.h>
#include <fstream>

traact::gui::TraactGuiApp::TraactGuiApp(const std::string &configFile) : config_file_(configFile) {
    facade::DefaultFacade facade;


    available_patterns_ = facade.GetAllAvailablePatterns();
    recent_files_.push_back("foo1");
    recent_files_.push_back("foo2");

}

std::vector<std::string> & traact::gui::TraactGuiApp::GetRecentFiles() {
    return recent_files_;
}

void traact::gui::TraactGuiApp::OpenFile(std::string file) {
    recent_files_.insert(recent_files_.begin(), file);
    if(recent_files_.size() > 5)
        recent_files_.reserve(5);

    auto loaded_pattern_graph_ptr = std::make_shared<pattern::instance::GraphInstance>();
    nlohmann::json json_graph;
    std::ifstream graph_file;
    graph_file.open(file);
    graph_file >> json_graph;
    graph_file.close();
    ns::from_json(json_graph, *loaded_pattern_graph_ptr);
    loaded_graphs_[file] = loaded_pattern_graph_ptr;
}

std::vector<std::string> traact::gui::TraactGuiApp::OpenFiles() {

    return open_files_;
}

void traact::gui::TraactGuiApp::CloseFile(std::string file) {


}

void traact::gui::TraactGuiApp::CloseAll() {

}

std::vector<traact::pattern::Pattern::Ptr> traact::gui::TraactGuiApp::GetAvailablePatterns() {
    return available_patterns_;
}

const std::map<std::string, traact::pattern::instance::GraphInstance::Ptr> traact::gui::TraactGuiApp::GetGraphInstances() {
    return loaded_graphs_;
}

traact::gui::TraactGuiApp::~TraactGuiApp() {

}

void traact::gui::TraactGuiApp::NewFile() {

}
