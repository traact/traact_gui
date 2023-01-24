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

    if(!data.isInputValid(port_index_)){
        return;
    }

    const auto& image = data.getInput<vision::ImageHeader>(port_index_).value();
    const auto& image_header = data.getInputHeader<vision::ImageHeader>(port_index_);

    if(data.getTimestamp() < last_image_upload_){
        SPDLOG_ERROR("image upload of older image last: {0} current: {1}", last_image_upload_, data.getTimestamp());
        return;
    }
    last_image_upload_ = data.getTimestamp();
    ++upload_count_;

    auto data_size = image_header.width * image_header.height * image_header.channels * getBytes(image_header.base_type);
    auto opengl_internal_format = getOpenGlInternalFormat(image_header);
    auto opengl_format = getOpenGlFormat(image_header);
    if (!init_texture_) {


        checkOpenGLErrors(glGenTextures(1, &texture_));
        checkOpenGLErrors(glBindTexture(GL_TEXTURE_2D, texture_));
        checkOpenGLErrors(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR));
        checkOpenGLErrors(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR));
        //checkOpenGLErrors(glBindTexture(GL_TEXTURE_2D, 0));

        checkOpenGLErrors(glGenBuffers(1, &pbo_id_));
        checkOpenGLErrors(glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo_id_));
        //checkOpenGLErrors(glBufferData(GL_PIXEL_UNPACK_BUFFER, data_size, 0, GL_STREAM_DRAW));
        //checkOpenGLErrors(glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0));

        SPDLOG_INFO("initialize with image component info: internal format {0} width {1} height {2} format {3}",opengl_internal_format, image_header.width, image_header.height, opengl_format);

        renderer_->setImageSize(ImVec2(image_header.width, image_header.height), window_name_);

        init_texture_ = true;
    } else {
        checkOpenGLErrors(glBindTexture(GL_TEXTURE_2D, texture_));
        checkOpenGLErrors(glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo_id_));

    }

    checkOpenGLErrors(glBufferData(GL_PIXEL_UNPACK_BUFFER, data_size, image.data, GL_STREAM_DRAW));

//    checkOpenGLErrors(glTexImage2D(GL_TEXTURE_2D, 0, opengl_internal_format, image_header.width, image_header.height, 0,
//                                   opengl_format, GL_UNSIGNED_BYTE, 0));
    checkOpenGLErrors(glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, image_header.width, image_header.height, 0,
                 getOpenGl(image_header.pixel_format), getOpenGl(image_header.base_type), 0));

    checkOpenGLErrors(glBindTexture(GL_TEXTURE_2D, 0));
    checkOpenGLErrors(glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0));
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