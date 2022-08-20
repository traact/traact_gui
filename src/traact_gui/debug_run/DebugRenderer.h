/** Copyright (C) 2022  Frieder Pankratz <frieder.pankratz@gmail.com> **/

#ifndef TRAACT_GUI_SRC_TRAACT_GUI_DEBUG_RUN_DEBUGRENDERER_H_
#define TRAACT_GUI_SRC_TRAACT_GUI_DEBUG_RUN_DEBUGRENDERER_H_

#include <imgui.h>
#define IMGUI_DEFINE_MATH_OPERATORS
#include <imgui_internal.h>
#include "external/imgui_misc/imgui_stdlib.h"

#include <traact/traact.h>
#include <traact/vision.h>
#include <traact/util/Semaphore.h>

#include "DebugRenderComponent.h"
#include "traact_gui/debug_run/DebugSceneWindow.h"



namespace traact::gui {



class DebugRenderer {

 public:
    DebugRenderer();
    void draw();
    void configureInstance(const pattern::instance::PatternInstance &pattern_instance);
    bool processTimePoint(traact::buffer::ComponentBuffer &data);
    void setImageSize(ImVec2 image_size, const std::string &window_name);
    void setImageRenderSize(ImVec2 image_size, const std::string &window_name);
    ImVec2 getScale(const std::string& window_name);
    void setCameraCalibration(const vision::CameraCalibration& calibration, const std::string& window_name);
    const vision::CameraCalibration& getCameraCalibration(const std::string& window_name);

 private:
    std::map<std::string, std::optional<ImVec2>> render_size_{};
    std::map<std::string, std::optional<ImVec2>> image_size_{};
    std::map<std::string, vision::CameraCalibration> camera_calibration_{};

    std::mutex command_lock_;
    std::map<std::string, std::vector<std::shared_ptr<DebugRenderComponent> > > render_components_;
    std::map<std::string, std::vector<RenderCommand> > additional_commands_;
    std::map<std::string, std::vector<RenderCommand> > render_commands_;
    WaitForInit additional_commands_processed_;

    std::shared_ptr<DebugSceneWindow> scene_window_;




    void addWindow(const std::vector<std::string> &port_segmented, pattern::instance::PortInstance::ConstPtr const &port);
    void addWindowImage(const std::string &window_name,
                        const std::vector<std::string> &port_segmented,
                        pattern::instance::PortInstance::ConstPtr const &port);
    void addScene(const std::vector<std::string> &port_segmented,
                  pattern::instance::PortInstance::ConstPtr const &port);
    void addWindowPose(const std::string &window_name,
                       const std::vector<std::string> &ports_segmented,
                       pattern::instance::PortInstance::ConstPtr const &port);
    void addWindowCalibration(const std::string &window_name,
                              const std::vector<std::string> &port_segmented,
                              pattern::instance::PortInstance::ConstPtr const &port);
};

} // traact

#endif //TRAACT_GUI_SRC_TRAACT_GUI_DEBUG_RUN_DEBUGRENDERER_H_
