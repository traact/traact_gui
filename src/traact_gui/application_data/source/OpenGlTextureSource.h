/** Copyright (C) 2022  Frieder Pankratz <frieder.pankratz@gmail.com> **/

#ifndef TRAACT_GUI_SRC_TRAACT_GUI_APPLICATION_DATA_SOURCE_OPENGLTEXTURESOURCE_H_
#define TRAACT_GUI_SRC_TRAACT_GUI_APPLICATION_DATA_SOURCE_OPENGLTEXTURESOURCE_H_

#include "traact_gui/application_data/DataPort.h"
#include <GL/gl.h>
#include <cuda_runtime.h>

namespace traact::application_data::source {

class OpenGlTextureSource : public DataPort {
 public:
    OpenGlTextureSource(ApplicationData *application_data, pattern::instance::PortInstance::ConstPtr const &port);
    virtual bool processTimePoint(traact::buffer::ComponentBuffer &data) override;

    void bind(int tex_id);
    int getWidth() const;
    int getHeight() const;
    int getStride() const;
    GLuint getOpenglTexture() const;
    GLint getOpenglInternalFormat() const;
    GLint getOpenglFormat() const;
 private:

    GLuint opengl_texture_;
    GLint opengl_internal_format_;
    GLint opengl_format_;
    int width_;
    int height_;
    int stride_;
    cudaGraphicsResource *cuda_resource_;
};

} // traact

#endif //TRAACT_GUI_SRC_TRAACT_GUI_APPLICATION_DATA_SOURCE_OPENGLTEXTURESOURCE_H_
