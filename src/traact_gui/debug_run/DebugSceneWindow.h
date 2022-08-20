/** Copyright (C) 2022  Frieder Pankratz <frieder.pankratz@gmail.com> **/

#ifndef TRAACT_GUI_SRC_TRAACT_GUI_DEBUG_RUN_DEBUGSCENEWINDOW_H_
#define TRAACT_GUI_SRC_TRAACT_GUI_DEBUG_RUN_DEBUGSCENEWINDOW_H_

#include "DebugRenderComponent.h"
#include "scene/Window.h"

#include "traact_gui/application_data/ApplicationData.h"

namespace traact::gui {

class DebugSceneWindow : public DebugRenderComponent{
 public:
    DebugSceneWindow(const std::string &window_name,
                     DebugRenderer *renderer);
    ~DebugSceneWindow() override;

    void update(buffer::ComponentBuffer &buffer, std::vector<RenderCommand> &additional_commands) override;

    void addDebugObject(const std::vector<std::string> &port_segmented,
                        pattern::instance::PortInstance::ConstPtr const &port);

 private:
    scene::Window window_;
    application_data::ApplicationData traact_app_data_;


    void addPointCloud(std::shared_ptr<scene::Object> scene_object,
    const std::vector<std::string> &port_segmented,
        pattern::instance::PortInstance::ConstPtr const &port);

    void addPose(scene::Object::SharedPtr object,
                 const std::vector<std::string> &port_segmented,
                 pattern::instance::PortInstance::ConstPtr const &port);
};

} // traact

#endif //TRAACT_GUI_SRC_TRAACT_GUI_DEBUG_RUN_DEBUGSCENEWINDOW_H_
