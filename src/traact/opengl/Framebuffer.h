/** Copyright (C) 2022  Frieder Pankratz <frieder.pankratz@gmail.com> **/

#ifndef TRAACT_GUI_SRC_TRAACT_OPENGL_FRAMEBUFFER_H_
#define TRAACT_GUI_SRC_TRAACT_OPENGL_FRAMEBUFFER_H_

#include <GL/gl.h>

namespace traact::opengl {

class Framebuffer {
 public:
    Framebuffer() = default;
    Framebuffer(int width, int height);
    void init();
    void bind();
    void resetBind();
    int getWidth() const;
    int getHeight() const;
    bool isInitialized() const;
    GLuint getFrameBuffer() const;
    GLuint getRenderTexture() const;
    GLuint getDepthBuffer() const;
    void setViewPort(int width, int height);

 private:
    int width_{1280};
    int height_{720};
    bool initialized_{false};
    GLuint frame_buffer_ = 0;
    GLuint render_texture_;
    GLuint depth_buffer_;

};

} // traact

#endif //TRAACT_GUI_SRC_TRAACT_OPENGL_FRAMEBUFFER_H_
