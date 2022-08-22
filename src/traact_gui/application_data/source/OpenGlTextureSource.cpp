/** Copyright (C) 2022  Frieder Pankratz <frieder.pankratz@gmail.com> **/

#include <GL/glew.h>
#include "OpenGlTextureSource.h"
#include <traact/vision.h>
#include <cuda_gl_interop.h>
#include "traact_gui/debug_run/traact_opengl.h"
#include "traact_gui/application_data/application_data.h"

namespace traact::application_data::source {
OpenGlTextureSource::OpenGlTextureSource(ApplicationData *application_data,
                                         pattern::instance::PortInstance::ConstPtr const &port) : DataPort(
    application_data,
    port) {

}

bool OpenGlTextureSource::processTimePoint(traact::buffer::ComponentBuffer &data) {

    if(!data.isInputValid(port_index_)){
        return true;
    }

    const auto &cuda_image = data.getInput<vision::GpuImageHeader>(port_index_).value();
    const auto &image_header = data.getInputHeader<vision::GpuImageHeader>(port_index_);

    if (!initialized_) {
        opengl_internal_format_ = getOpenGlInternalFormat(image_header);
        opengl_format_ = getOpenGlFormat(image_header);
        width_ = image_header.width;
        height_ = image_header.height;
        stride_ = image_header.stride;
        checkOpenGLErrors(glGenTextures(1, &opengl_texture_));
        checkOpenGLErrors(glBindTexture(GL_TEXTURE_2D, opengl_texture_));
        checkOpenGLErrors(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR));
        checkOpenGLErrors(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR));
        SPDLOG_INFO("initialize with texture info: internal format {0} width {1} height {2} format {3}",opengl_internal_format_, image_header.width, image_header.height, opengl_format_);
        checkOpenGLErrors(glTexImage2D(GL_TEXTURE_2D, 0, opengl_internal_format_, image_header.width, image_header.height, 0,
                                       opengl_format_, GL_UNSIGNED_BYTE, 0));
        checkCudaErrors(cudaGraphicsGLRegisterImage(&cuda_resource_,
                                                    opengl_texture_,
                                                    GL_TEXTURE_2D,
                                                    cudaGraphicsRegisterFlagsWriteDiscard));
        checkOpenGLErrors(glBindTexture(GL_TEXTURE_2D, 0));

        initialized_ = true;
    }

    auto stream = application_data_->getCudaStream();

    checkCudaErrors(cudaGraphicsMapResources(1, &cuda_resource_, stream));

    cudaArray *texture_ptr;
    checkCudaErrors(cudaGraphicsSubResourceGetMappedArray(&texture_ptr, cuda_resource_, 0, 0));
    checkCudaErrors(cudaMemcpy2DToArrayAsync(texture_ptr, 0, 0, cuda_image.cudaPtr(),
                                             cuda_image.step, cuda_image.step,
                                             cuda_image.rows, cudaMemcpyDeviceToDevice, stream));

    checkCudaErrors(cudaGraphicsUnmapResources(1, &cuda_resource_, stream));

    return true;
}
void OpenGlTextureSource::bind(int tex_id) {

    if(!initialized_){
        return;
    }

    checkOpenGLErrors(glActiveTexture(GL_TEXTURE0+tex_id));
    checkOpenGLErrors(glBindTexture(GL_TEXTURE_2D, opengl_texture_));
    checkOpenGLErrors(glBindImageTexture(tex_id,
                                         opengl_texture_,
                                         0,
                                         GL_FALSE,
                                         0,
                                         GL_READ_ONLY,
                                         opengl_internal_format_));

}
int OpenGlTextureSource::getWidth() const {
    return width_;
}
int OpenGlTextureSource::getHeight() const {
    return height_;
}
int OpenGlTextureSource::getStride() const {
    return stride_;
}
GLuint OpenGlTextureSource::getOpenglTexture() const {
    return opengl_texture_;
}
GLint OpenGlTextureSource::getOpenglInternalFormat() const {
    return opengl_internal_format_;
}
GLint OpenGlTextureSource::getOpenglFormat() const {
    return opengl_format_;
}

}

// traact