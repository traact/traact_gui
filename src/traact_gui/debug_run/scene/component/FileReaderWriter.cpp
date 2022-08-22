/** Copyright (C) 2022  Frieder Pankratz <frieder.pankratz@gmail.com> **/

#include <traact/spatial_convert.h>
#include "FileReaderWriter.h"
#include "traact_gui/debug_run/traact_opengl.h"

namespace traact {
gui::scene::component::FileReaderWriter::FileReaderWriter(const std::shared_ptr<Object> &object,
                                                          const std::string &name) : Component(object, name) {}
void gui::scene::component::FileReaderWriter::setReader(const std::shared_ptr<traact::component::FileReaderWriterRead<
    spatial::Pose6DHeader>> &reader) {
    reader_ = reader;
}
void gui::scene::component::FileReaderWriter::setWriter(const std::shared_ptr<traact::component::FileReaderWriterWrite<
    spatial::Pose6DHeader>> &writer) {
    writer_ = writer;
}
void gui::scene::component::FileReaderWriter::update() {
    if(reader_ && current_mode_ == Mode::Use){
        useCalibration();
    }

}
void gui::scene::component::FileReaderWriter::useCalibration() {
    auto traact_pose = reader_->getValue();
    Eigen::Affine3f opengl_pose = spatial::convert<spatial::TraactCoordinateSystem, spatial::OpenGLCoordinateSystem>(traact_pose);
    transform_->setLocalPose(eigen2glm(opengl_pose.matrix()));
}
void gui::scene::component::FileReaderWriter::drawGui() {
    ImGui::BeginChild("#Calibration");

    ImGui::BeginDisabled(reader_ == nullptr);
    if (ImGui::RadioButton("#Use", current_mode_ == Mode::Use)){
        current_mode_ = Mode::Use;
    }
    ImGui::EndDisabled();

    ImGui::SameLine();
    ImGui::BeginDisabled(writer_ == nullptr);
    if (ImGui::RadioButton("#Modify", current_mode_ == Mode::Modify)){
        current_mode_ = Mode::Modify;
    }
    ImGui::EndDisabled();

    ImGui::BeginDisabled(current_mode_ != Mode::Modify);
    if(ImGui::Button("Save")){
        saveCalibration();
    }
    ImGui::EndDisabled();
    ImGui::EndChild();
}
void gui::scene::component::FileReaderWriter::saveCalibration() {
    auto eigen_opengl = Eigen::Map<const Eigen::Matrix4f>(glm::value_ptr(transform_->getLocalPose()));
    auto opengl_pose = Eigen::Affine3f(eigen_opengl);
    spatial::Pose6D traact_pose = spatial::convert<spatial::OpenGLCoordinateSystem, spatial::TraactCoordinateSystem>(traact_pose);
    writer_->setValue(traact_pose);
}
} // traact