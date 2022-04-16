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
#include <traact_gui/DataflowFile.h>
#include <util/fileutil.h>
#include <traact/util/RingBuffer.h>

namespace traact::gui {

    class TraactGuiApp {
    public:
        TraactGuiApp(const std::string &configFile);

        ~TraactGuiApp();

        void OnFrame();

        void NewFile();
        void NewFile(const std::string& dataflow_json);
        void OpenFile(fs::path file);
        const std::vector<std::string> & OpenFiles();
        void CloseFile(std::string file);
        void CloseAll();


    private:
        std::string config_file_;
        buffers::ring_buffer<std::string, 5> recent_files_;
        std::vector<std::string> open_files_;

        std::vector<pattern::Pattern::Ptr> available_patterns_;

        std::vector<DataflowFile*> dataflow_files_;
        DataflowFile* current_dataflow_{nullptr};
        DataflowFile* pending_dataflow_{nullptr};

        void MenuBar();

        void DrawLeftPanel(int width, int height);
        void DrawRightPanel(int width, int height);

        void DrawPatternPanel();
        void DrawDetailsPanel();

        void SaveConfig();
        void LoadConfig();



    };
}



#endif //TRAACTMULTI_TRAACTGUIAPP_H
