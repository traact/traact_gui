#include <application.h>

#include <imgui_node_editor.h>



#define IMGUI_DEFINE_MATH_OPERATORS
#include <imgui_internal.h>

#include <string>
#include <vector>
#include <map>
#include <algorithm>
#include <utility>
#include <imgui_node_editor_internal.h>
#include "TraactGuiApp.h"
#include "windows/PatternTree.h"

#include <GLFW/glfw3.h>
#include <GLFW/glfw3native.h>



namespace ed = ax::NodeEditor;

using namespace ax;
using namespace traact::gui;

static TraactGuiApp traact_app("traact_gui.json");
static PatternTree pattern_tree(&traact_app);

static inline ImRect ImGui_GetItemRect()
{
    return ImRect(ImGui::GetItemRectMin(), ImGui::GetItemRectMax());
}

static inline ImRect ImRect_Expanded(const ImRect& rect, float x, float y)
{
    auto result = rect;
    result.Min.x -= x;
    result.Min.y -= y;
    result.Max.x += x;
    result.Max.y += y;
    return result;
}


const char* Application_GetName()
{
    return "Traact GUI";
}

void Application_Initialize()
{
    ed::Config config;

    config.SettingsFile = "traact_gui.json";


//    config.LoadNodeSettings = [](ed::NodeId nodeId, char* data, void* userPointer) -> size_t
//    {
//        auto node = FindNode(nodeId);
//        if (!node)
//            return 0;
//
//        if (data != nullptr)
//            memcpy(data, node->State.data(), node->State.size());
//        return node->State.size();
//    };
//
//    config.SaveNodeSettings = [](ed::NodeId nodeId, const char* data, size_t size, ed::SaveReasonFlags reason, void* userPointer) -> bool
//    {
//        auto node = FindNode(nodeId);
//        if (!node)
//            return false;
//
//        node->State.assign(data, size);
//
//        TouchNode(nodeId);
//
//        return true;
//    };


    auto& io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;


}

void Application_Finalize()
{

}


void Application_Frame()
{

    auto& io = ImGui::GetIO();


    if (ImGui::BeginMainMenuBar())
    {
        if (ImGui::BeginMenu("File"))
        {
            if (ImGui::MenuItem("New")) {}
            if (ImGui::MenuItem("Open", "Ctrl+O")) {}
            if (ImGui::BeginMenu("Open Recent"))
            {
                for(const auto& file_name : traact_app.GetRecentFiles()){
                    if(ImGui::MenuItem(file_name.c_str())){
                        traact_app.OpenFile(file_name);
                    }
                }
                ImGui::EndMenu();
            }
            if (ImGui::MenuItem("Save", "Ctrl+S")) {}
            if (ImGui::MenuItem("Save As..")) {}

            ImGui::Separator();

            if (ImGui::MenuItem("Quit", "Alt+F4")) {}

            ImGui::EndMenu();
        }
        if (ImGui::BeginMenu("Edit"))
        {
            if (ImGui::MenuItem("Undo", "CTRL+Z", false, false)) {}
            if (ImGui::MenuItem("Redo", "CTRL+Y", false, false)) {}
//            ImGui::Separator();
//            if (ImGui::MenuItem("Cut", "CTRL+X")) {}
//            if (ImGui::MenuItem("Copy", "CTRL+C")) {}
//            if (ImGui::MenuItem("Paste", "CTRL+V")) {}
            ImGui::EndMenu();
        }
        ImGui::EndMainMenuBar();
    }

    pattern_tree.Draw();


    ImGui::ShowDemoWindow();




    //ImGui::ShowMetricsWindow();
}


