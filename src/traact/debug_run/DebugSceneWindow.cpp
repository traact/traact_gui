/** Copyright (C) 2022  Frieder Pankratz <frieder.pankratz@gmail.com> **/
#include <GL/glew.h>
//#include "magnum/Window.h"
#include "DebugSceneWindow.h"

#include "scene/component/RenderPointCloud.h"
#include "scene/component/RenderCoordinateSystem.h"
#include "scene/component/PoseSource.h"
#include "scene/component/IdentityRotation.h"
#include "scene/component/FileReaderWriter.h"

#include "traact/application_data/application_data.h"
#include <traact/component/generic/FileReaderWriter.h>

namespace traact::gui {
DebugSceneWindow::DebugSceneWindow(const std::string &window_name,
                                   DebugRenderer *renderer) : DebugRenderComponent(100, 0, "invalid", window_name, renderer) {


    //magnum_window_ = std::make_unique<traact::gui::magnum::Window>();
    //magnum_window_->init();
    render_command_ = [this]() {
        window_.update();
        window_.draw();
//        magnum_window_->update();
//        magnum_window_->draw();
    };


    auto origin_object = window_.addObject("origin");
    origin_object->addComponent<scene::component::RenderCoordinateSystem>("coordinate_frame" );

    auto& camera_transform = window_.getMainCamera()->getTransform();
    auto& camera_object = window_.getMainCamera()->getObject();

    auto hud_object = window_.addObject("hud");
    auto hud_coordinate_frame = hud_object->addComponent<scene::component::RenderCoordinateSystem>("coordinate_frame" );
    hud_object->addComponent<scene::component::IdentityRotation>("IdentityRotation" );

    hud_coordinate_frame->scale_ = 0.03;
    hud_object->getTransform()->setParent(camera_transform);
    glm::mat4 hud_pose(1.0);
    hud_pose[3].x = 0.16;
    hud_pose[3].y = 0.09;
    hud_pose[3].z = -0.25;
    hud_object->getTransform()->setLocalPose(hud_pose);



}

DebugSceneWindow::~DebugSceneWindow() {
    traact_app_data_.destroy();

}

void DebugSceneWindow::update(buffer::ComponentBuffer &buffer, std::vector<RenderCommand> &additional_commands) {
    window_.update();

    additional_commands.emplace_back([&app_data = traact_app_data_, &buffer](){
       app_data.processTimePoint(buffer);
    });
}

void DebugSceneWindow::addDebugObject(const std::vector<std::string> &port_segmented,
                                      pattern::instance::PortInstance::ConstPtr const &port) {

    auto object_id = port_segmented[1];
    auto object_name = fmt::format("{0}", object_id);

    auto object = window_.getObject(object_name);

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
    auto source = traact_app_data_.addDataPort<application_data::source::OpenGlTextureSource>(port);

    if(port_segmented[3] == "vertex"){
        component->setVertexSource(source);
    } else if(port_segmented[3] == "color"){
        component->setColorSource(source);
    } else {
        SPDLOG_ERROR("unsupported scene point cloud data: {0}", port_segmented[3]);
    }

    if(port->getDataType() == vision::GpuImageHeader::NativeTypeName){

    }
}

void DebugSceneWindow::addPose(scene::Object::SharedPtr object,
                               const std::vector<std::string> &port_segmented,
                               pattern::instance::PortInstance::ConstPtr const &port) {

    auto parent_object = window_.getObject(port_segmented[3]);

    auto component = object->getComponent<scene::component::PoseSource>("pose_source");
    auto pose_source = traact_app_data_.addDataPort<application_data::PoseSource >(port);
    component->setSource(pose_source);

    object->getTransform()->setParent(parent_object->getTransform());

    auto coordinate_system = object->getComponent<scene::component::RenderCoordinateSystem>("coordinate_frame" );
    coordinate_system->scale_ = 0.09;


}

template<>
void DebugSceneWindow::addDebugObject<std::shared_ptr<component::FileReaderWriterRead<spatial::Pose6DHeader>>>(const std::vector<std::string> &name_segmented, std::shared_ptr<component::FileReaderWriterRead<spatial::Pose6DHeader>>&pattern_instance) {

    if(name_segmented.size() != 4){
        SPDLOG_ERROR("naming of FileReaderWriterRead component must follow the scheme: scene_{targetID}_calibrationRead_{sourceID}");
        return;
    }
    auto& target = name_segmented[1];
    auto& source = name_segmented[3];
    auto object = window_.getObject(target);
    auto parent_object = window_.getObject(source);

    auto component = object->getComponent<scene::component::FileReaderWriter>("file_reader_writer");
    component->setReader(pattern_instance);

    object->getTransform()->setParent(parent_object->getTransform());

    auto coordinate_system = object->getComponent<scene::component::RenderCoordinateSystem>("coordinate_frame" );
    coordinate_system->scale_ = 0.09;
}

template<>
void DebugSceneWindow::addDebugObject<std::shared_ptr<component::FileReaderWriterWrite<spatial::Pose6DHeader>>>(const std::vector<std::string> &name_segmented, std::shared_ptr<component::FileReaderWriterWrite<spatial::Pose6DHeader>>&pattern_instance) {

    if(name_segmented.size() != 4){
        SPDLOG_ERROR("naming of FileReaderWriterRead component must follow the scheme: scene_{targetID}_calibrationRead_{sourceID}");
        return;
    }
    auto& target = name_segmented[1];
    auto& source = name_segmented[3];
    auto object = window_.getObject(target);
    auto parent_object = window_.getObject(source);

    auto component = object->getComponent<scene::component::FileReaderWriter>("file_reader_writer");
    component->setWriter(pattern_instance);

    object->getTransform()->setParent(parent_object->getTransform());

    auto coordinate_system = object->getComponent<scene::component::RenderCoordinateSystem>("coordinate_frame" );
    coordinate_system->scale_ = 0.09;
}


} // traact