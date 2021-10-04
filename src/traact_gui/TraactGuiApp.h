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

#ifndef TRAACTMULTI_TRAACTGUIAPP_H
#define TRAACTMULTI_TRAACTGUIAPP_H

#include <vector>
#include <string>
#include <map>
#include <traact/pattern/instance/GraphInstance.h>

namespace traact::gui {

    class TraactGuiApp {
    public:
        TraactGuiApp(const std::string &configFile);

        virtual ~TraactGuiApp();

        std::vector<std::string> & GetRecentFiles();

        void NewFile();
        void OpenFile(std::string file);
        std::vector<std::string> OpenFiles();
        void CloseFile(std::string file);
        void CloseAll();

        std::vector<pattern::Pattern::Ptr> GetAvailablePatterns();
        const std::map<std::string, pattern::instance::GraphInstance::Ptr> GetGraphInstances();


    private:
        std::string config_file_;
        std::vector<std::string> recent_files_;
        std::vector<std::string> open_files_;
        std::map<std::string, std::string> loaded_graph_files_;
        std::map<std::string, pattern::instance::GraphInstance::Ptr> loaded_graphs_;
        std::vector<pattern::Pattern::Ptr> available_patterns_;

    };
}



#endif //TRAACTMULTI_TRAACTGUIAPP_H
