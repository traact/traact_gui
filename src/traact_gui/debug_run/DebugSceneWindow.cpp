/** Copyright (C) 2022  Frieder Pankratz <frieder.pankratz@gmail.com> **/

#include "DebugSceneWindow.h"
#include "scene/component/RenderPointCloud.h"
#include "scene/component/RenderCoordinateSystem.h"
#include "scene/component/PoseSource.h"

#include "traact_gui/application_data/application_data.h"

namespace traact::gui {
DebugSceneWindow::DebugSceneWindow(const std::string &window_name,
                                   DebugRenderer *renderer) : DebugRenderComponent(100, 0, "invalid", window_name, renderer) {

    render_command_ = [this]() {
        window_.draw();
    };


    auto origin_object = window_.addObject("origin");
    origin_object->addComponent<scene::component::RenderCoordinateSystem>("coordinate_frame" );


}
void DebugSceneWindow::update(buffer::ComponentBuffer &buffer, std::vector<RenderCommand> &additional_commands) {
    window_.update(buffer, additional_commands);

    additional_commands.emplace_back([&app_data = traact_app_data_, &buffer](){
       app_data.processTimePoint(buffer);
    });
}

void DebugSceneWindow::addDebugObject(const std::vector<std::string> &port_segmented,
                                      pattern::instance::PortInstance::ConstPtr const &port) {

    auto object_id = port_segmented[1];
    auto object_name = fmt::format("{0}", object_id);

    auto object = window_.findObject(object_name);
    if(!object){
        object = window_.addObject(object_name);
    }


    if(port_segmented[2] == "pointCloud"){
        addPointCloud(object, port_segmented, port);
    } else if(port_segmented[2] == "pose"){
        addPose(object, port_segmented, port);
    } else{
        SPDLOG_ERROR("unsupported scene purpose: {0}", port_segmented[2]);
    }

}

void DebugSceneWindow::addPointCloud(std::shared_ptr<scene::Object> scene_object,
                           const std::vector<std::string> &port_segmented,
                           pattern::instance::PortInstance::ConstPtr const &port) {

    auto component = scene_object->getComponent<scene::component::RenderPointCloud>("cloud_render");
    if(port_segmented[3] == "vertex"){
        component->setVertexPort(port->getPortIndex());
    } else if(port_segmented[3] == "color"){
        component->setColorPort(port->getPortIndex());
    } else {
        SPDLOG_ERROR("unsupported scene point cloud data: {0}", port_segmented[3]);
    }

    if(port->getDataType() == vision::GpuImageHeader::NativeTypeName){

    }
}
void DebugSceneWindow::addPose(scene::Object::SharedPtr object,
                               const std::vector<std::string> &port_segmented,
                               pattern::instance::PortInstance::ConstPtr const &port) {

    auto parent_object = window_.findObject(port_segmented[3]);
    if(!parent_object){
        parent_object = window_.addObject(port_segmented[3]);
    }

    auto component = object->getComponent<scene::component::PoseSource>("pose_source");
    auto pose_source = traact_app_data_.addDataPort<application_data::PoseSource >(port);
    component->setPosePort(port->getPortIndex());

    object->getTransform()->setParent(parent_object->getTransform());

    auto coordinate_system = object->getComponent<scene::component::RenderCoordinateSystem>("coordinate_frame" );
    coordinate_system->scale_ = 0.09;


}

DebugSceneWindow::~DebugSceneWindow() {
    traact_app_data_.destroy();

}
} // traact