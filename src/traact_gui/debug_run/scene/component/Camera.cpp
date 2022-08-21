/** Copyright (C) 2022  Frieder Pankratz <frieder.pankratz@gmail.com> **/

#include <GLFW/glfw3.h>
#include "Camera.h"
#include <spdlog/spdlog.h>
#include <imgui.h>
namespace traact::gui::scene::component {
Camera::Camera(const std::shared_ptr<Object> &object,std::string name) : Component(object, std::move(name)) {
    updateProjection();
    transform_->setLocalPose(glm::inverse(getViewMatrix()));
}
void Camera::updateProjection() { projection_ = glm::perspective(glm::radians(fov_), 16.0f/9.0f, 0.1f, 100.0f); }

void Camera::moveForward() {
    camera_pos_ += getMoveSpeed() * camera_front_;
}
void Camera::moveBackward() {
    camera_pos_ -= getMoveSpeed() * camera_front_;
}
void Camera::moveShiftLeft() {
    camera_pos_ -= glm::normalize(glm::cross(camera_front_, camera_up_)) * getMoveSpeed();
}
void Camera::moveShiftRight() {
    camera_pos_ += glm::normalize(glm::cross(camera_front_, camera_up_)) * getMoveSpeed();
}
void Camera::moveUp() {
    camera_pos_ += getMoveSpeed() * camera_up_;
}
void Camera::moveDown() {
    camera_pos_ -= getMoveSpeed() * camera_up_;
}

float Camera::getMoveSpeed() const {
    return delta_time_ * move_speed_;
}
glm::mat4 Camera::getViewMatrix() const {
    return view_;
}
void Camera::mouseRotate(float dx, float dy) {

    camera_yaw_   += dx * rotate_speed_;
    camera_pitch_ += -dy * rotate_speed_;

    if(camera_pitch_ > 89.0f)
        camera_pitch_ =  89.0f;
    if(camera_pitch_ < -89.0f)
        camera_pitch_ = -89.0f;

    glm::vec3 direction;
    direction.x = cos(glm::radians(camera_yaw_)) * cos(glm::radians(camera_pitch_));
    direction.y = sin(glm::radians(camera_pitch_));
    direction.z = sin(glm::radians(camera_yaw_)) * cos(glm::radians(camera_pitch_));
    camera_front_ = glm::normalize(direction);

    auto camera_right = glm::normalize(glm::cross(camera_front_, glm::vec3(0,1,0)));  // normalize the vectors, because their length gets closer to 0 the more you look up or down which results in slower movement.
    camera_up_    = glm::normalize(glm::cross(camera_right, camera_front_));

}
void Camera::updateMovement() {
    float current = glfwGetTime();
    delta_time_ = current - last_frame_;
    last_frame_ = current;

    ImGuiIO &io = ImGui::GetIO();

    if (ImGui::IsWindowHovered() && io.MouseDown[ImGuiMouseButton_Right]) {

        if (ImGui::IsKeyDown(ImGuiKey_W)) {
            moveForward();
        }
        if (ImGui::IsKeyDown(ImGuiKey_S)) {
            moveBackward();
        }
        if(ImGui::IsKeyDown(ImGuiKey_D)){
            moveShiftRight();
        }
        if(ImGui::IsKeyDown(ImGuiKey_A)){
            moveShiftLeft();
        }
        if(ImGui::IsKeyDown(ImGuiKey_E)){
            moveUp();
        }
        if(ImGui::IsKeyDown(ImGuiKey_Q)){
            moveDown();
        }

        if(io.MouseDelta.x != 0 || io.MouseDelta.y != 0){
            mouseRotate(io.MouseDelta.x, io.MouseDelta.y);
        }
    }

    view_ = glm::lookAt(camera_pos_, camera_pos_ + camera_front_, camera_up_);
}
glm::mat4 Camera::getProjectionMatrix()  const{
    return projection_;
}
glm::mat4 Camera::getVPMatrix() const {
    return projection_ * view_;
}

void Camera::draw() {
    updateMovement();
    updateProjection();
    transform_->setLocalPose(glm::inverse(getViewMatrix()));
}
void Camera::drawGui() {
    ImGui::SliderFloat("TranslationSpeed", &move_speed_, 0.0f, 1.0f);
    ImGui::SliderFloat("RotationSpeed", &rotate_speed_, 0.0f, 1.0f);
    ImGui::SliderFloat("FOV", &fov_, 15.0f, 90.0f);
}
const float *Camera::getViewMatrixPtr() const {
    return glm::value_ptr(view_);
}
const float *Camera::getProjectionMatrixPtr() const {
    return glm::value_ptr(projection_);
}

} // traact