/** Copyright (C) 2022  Frieder Pankratz <frieder.pankratz@gmail.com> **/

#include <GL/glew.h>
#include "ImageComponent.h"
#include "DebugRenderer.h"

namespace traact::gui {
ImageComponent::ImageComponent(int port_index,
                               const std::string &port_name,
                               const std::string &window_name,
                               DebugRenderer *renderer)
    : DebugRenderComponent(
    1000,
    port_index,
    port_name, window_name, renderer) {

    render_command_ = [this]() {
        draw();
    };
}
void ImageComponent::update(buffer::ComponentBuffer &buffer, std::vector<RenderCommand> &additional_commands) {

    if(buffer.isInputValid(port_index_)){
        additional_commands.emplace_back([this, &buffer](){
            uploadImage(buffer);
        });
    }
}

void ImageComponent::uploadImage(buffer::ComponentBuffer &data) {
    SPDLOG_TRACE("{0}: uploadImage {1}", port_name_, data.getTimestamp());

    const auto& image = data.getInput<vision::ImageHeader>(port_index_).value();
    const auto& image_header = data.getInputHeader<vision::ImageHeader>(port_index_);

    if(data.getTimestamp() < last_image_upload_){
        SPDLOG_ERROR("image upload of older image last: {0} current: {1}", last_image_upload_, data.getTimestamp());
        return;
    }
    last_image_upload_ = data.getTimestamp();
    ++upload_count_;

    auto data_size = image_header.width * image_header.height * image_header.channels * getBytes(image_header.base_type);
    if (!init_texture_) {
        glGenTextures(1, &texture_);
        glBindTexture(GL_TEXTURE_2D, texture_);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glBindTexture(GL_TEXTURE_2D, 0);

        glGenBuffers(1, &pbo_id_);
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo_id_);
        glBufferData(GL_PIXEL_UNPACK_BUFFER, data_size, 0, GL_STREAM_DRAW);
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);


        renderer_->setImageSize(ImVec2(image_header.width, image_header.height), window_name_);

        init_texture_ = true;
    }
    glBindTexture(GL_TEXTURE_2D, texture_);
    //glPixelStorei(GL_UNPACK_ROW_LENGTH, 0);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo_id_);
    glBufferData(GL_PIXEL_UNPACK_BUFFER, data_size, image.data, GL_STREAM_DRAW);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, image_header.width, image_header.height, 0,
                 getOpenGl(image_header.pixel_format), getOpenGl(image_header.base_type), 0);
    glBindTexture(GL_TEXTURE_2D, 0);

    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
}
void ImageComponent::draw() {
    ImGui::Begin(window_name_.c_str(), nullptr, ImGuiWindowFlags_NoDocking);
    ImVec2 avail_size = ImGui::GetContentRegionAvail();
    if (avail_size.x < 10) {
        avail_size.x = 320;
        avail_size.y = 180;
    }

    renderer_->setImageRenderSize(avail_size, window_name_);

    glBindTexture(GL_TEXTURE_2D, texture_);
    ImGui::Image(reinterpret_cast<void *>( static_cast<intptr_t>( texture_ )), avail_size);
    ImGui::End();
}
} // traact