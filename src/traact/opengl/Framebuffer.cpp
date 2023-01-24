/** Copyright (C) 2022  Frieder Pankratz <frieder.pankratz@gmail.com> **/

#include <GL/glew.h>

#include "Framebuffer.h"
#include "traact_opengl.h"

namespace traact {
opengl::Framebuffer::Framebuffer(int width, int height) : width_(width), height_(height) {}

void opengl::Framebuffer::init() {
    if (!initialized_) {

        checkOpenGLErrors(glGenFramebuffers(1, &frame_buffer_));
        checkOpenGLErrors(glGenTextures(1, &render_texture_));

        checkOpenGLErrors(glBindTexture(GL_TEXTURE_2D, render_texture_));
        checkOpenGLErrors(glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width_, height_, 0, GL_RGBA, GL_UNSIGNED_BYTE, 0));
        checkOpenGLErrors(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST));
        checkOpenGLErrors(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST));

        checkOpenGLErrors(glGenRenderbuffers(1, &depth_buffer_));
        checkOpenGLErrors(glBindRenderbuffer(GL_RENDERBUFFER, depth_buffer_));
        checkOpenGLErrors(glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT, width_, height_));

        initialized_ = true;


    }
}
void opengl::Framebuffer::bind() {
    glBindRenderbuffer(GL_RENDERBUFFER, depth_buffer_);
    glBindFramebuffer(GL_FRAMEBUFFER, frame_buffer_);

    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, depth_buffer_);
    glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, render_texture_, 0);

    GLenum DrawBuffers[1] = {GL_COLOR_ATTACHMENT0};
    glDrawBuffers(1, DrawBuffers);

    const GLenum frameBufferStatus = glCheckFramebufferStatus(GL_FRAMEBUFFER);
    if (frameBufferStatus != GL_FRAMEBUFFER_COMPLETE) {

    }
}
int opengl::Framebuffer::getWidth() const {
    return width_;
}
int opengl::Framebuffer::getHeight() const {
    return height_;
}
bool opengl::Framebuffer::isInitialized() const {
    return initialized_;
}
GLuint opengl::Framebuffer::getFrameBuffer() const {
    return frame_buffer_;
}
GLuint opengl::Framebuffer::getRenderTexture() const {
    return render_texture_;
}
GLuint opengl::Framebuffer::getDepthBuffer() const {
    return depth_buffer_;
}
void opengl::Framebuffer::setViewPort(int width, int height) {
    width_ = width;
    height_ = height;

}
void opengl::Framebuffer::resetBind() {
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glBindRenderbuffer(GL_RENDERBUFFER, 0);
}
} // traact