/** Copyright (C) 2022  Frieder Pankratz <frieder.pankratz@gmail.com> **/

#ifndef TRAACT_GUI_SRC_TRAACT_GUI_DEBUG_RUN_TRAACT_OPENGL_H_
#define TRAACT_GUI_SRC_TRAACT_GUI_DEBUG_RUN_TRAACT_OPENGL_H_

#include <traact/vision.h>
#include <GL/gl.h>
#include <GLFW/glfw3.h>
#include <spdlog/spdlog.h>
#include <glm/glm.hpp>
#include <glm/ext.hpp>
#include <Eigen/Dense>
#include <glm/matrix.hpp>

namespace traact {

static const constexpr GLint getOpenGl(vision::PixelFormat pixel_format) {
    switch (pixel_format) {
        case vision::PixelFormat::LUMINANCE:return GL_LUMINANCE;
        case vision::PixelFormat::RGB: return GL_RGB;
        case vision::PixelFormat::BGR: return GL_BGR;
        case vision::PixelFormat::RGBA: return GL_RGBA;
        case vision::PixelFormat::BGRA:return GL_BGRA;
        case vision::PixelFormat::DEPTH:return GL_R;
        case vision::PixelFormat::FLOAT:return GL_R;

        case vision::PixelFormat::YUV422:
        case vision::PixelFormat::YUV411:
        case vision::PixelFormat::RAW:
        case vision::PixelFormat::MJPEG:
        case vision::PixelFormat::UNKNOWN_PIXELFORMAT:return GL_INVALID_ENUM;
    }
}

static const constexpr GLint getOpenGl(BaseType data_type) {
    switch (data_type) {

        case INT_8: return GL_BYTE;
        case UINT_8:return GL_UNSIGNED_BYTE;
        case INT_16:return GL_SHORT;
        case UINT_16:return GL_UNSIGNED_SHORT;
        case INT_32:return GL_INT;
        case FLOAT_16:return GL_HALF_FLOAT;
        case FLOAT_32:return GL_FLOAT;
        case FLOAT_64:return GL_DOUBLE;
        case UNKNOWN: return GL_INVALID_ENUM;

    }
}

static void uploadImageOpenGL(const GLuint texture,
                              const GLvoid *data,
                              GLint internalFormat,
                              const vision::ImageHeader &header) {
    glBindTexture(GL_TEXTURE_2D, texture);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glPixelStorei(GL_UNPACK_ROW_LENGTH, 0);

    glTexImage2D(GL_TEXTURE_2D, 0, internalFormat, header.width, header.height, 0,
                 getOpenGl(header.pixel_format), getOpenGl(header.base_type), data);
}

static inline void drawCoordinateFrame(glm::mat4 world2model, float scale) {
    glMatrixMode(GL_MODELVIEW);
    glLoadMatrixf(glm::value_ptr(world2model));
    // x
    glColor3f(1.0, 0.0, 0.0);
    glBegin(GL_LINES);
    glVertex3f(-1.0 * scale, 0.0f, 0.0f);
    glVertex3f(1.0 * scale, 0.0f, 0.0f);

    glVertex3f(1.0 * scale, 0.0f, 0.0f);
    glVertex3f(0.75f * scale, 0.25f * scale, 0.0f);

    glVertex3f(1.0 * scale, 0.0f, 0.0f);
    glVertex3f(0.75f * scale, -0.25f * scale, 0.0f);
    glEnd();

    // y
    glColor3f(0.0, 1.0, 0.0);
    glBegin(GL_LINES);
    glVertex3f(0.0, -1.0f * scale, 0.0f);
    glVertex3f(0.0, 1.0f * scale, 0.0f);

    glVertex3f(0.0, 1.0f * scale, 0.0f);
    glVertex3f(0.25f * scale, 0.75f * scale, 0.0f);

    glVertex3f(0.0, 1.0f * scale, 0.0f);
    glVertex3f(-0.25f * scale, 0.75f * scale, 0.0f);
    glEnd();

    // z
    glColor3f(0.0, 0.0, 1.0);
    glBegin(GL_LINES);
    glVertex3f(0.0, 0.0f, -1.0f * scale);
    glVertex3f(0.0, 0.0f, 1.0f * scale);

    glVertex3f(0.0, 0.0f, 1.0f * scale);
    glVertex3f(0.0, 0.25f * scale, 0.75f * scale);

    glVertex3f(0.0, 0.0f, 1.0f * scale);
    glVertex3f(0.0, -0.25f * scale, 0.75f * scale);
    glEnd();
}



template<typename T, int m, int n>
inline glm::mat<m, n, float, glm::precision::highp> eigen2glm(const Eigen::Matrix<T, m, n>& eigen_matrix)
{
    glm::mat<m, n, float, glm::precision::highp> glm_matrix;
    for (int i = 0; i < m; ++i)
    {
        for (int j = 0; j < n; ++j)
        {
            glm_matrix[j][i] = eigen_matrix(i, j);
        }
    }
    return glm_matrix;
}

template<typename T, int m>
inline glm::vec<m, float, glm::precision::highp> eigen2glm(const Eigen::Matrix<T, m, 1>& eigen_vector)
{
    glm::vec<m, float, glm::precision::highp> glm_vector;
    for (int i = 0; i < m; ++i)
    {
        glm_vector[i] = eigen_vector(i);
    }
    return glm_vector;
}

}

#define checkCudaErrors(call)                                 \
  do {                                                        \
    cudaError_t err = call;                                   \
    if (err != cudaSuccess) {                                 \
      printf("CUDA error at %s %d: %s\n", __FILE__, __LINE__, \
             cudaGetErrorString(err));                        \
      exit(EXIT_FAILURE);                                     \
    }                                                         \
  } while (0)

#define checkOpenGLErrors(call)                                 \
    {                                                           \
        call;                                                   \
        auto error = glGetError();                              \
        if(error != GL_NO_ERROR){                               \
            SPDLOG_ERROR("OpenGL error {0}", error);\
        }                                                       \
    }

#endif //TRAACT_GUI_SRC_TRAACT_GUI_DEBUG_RUN_TRAACT_OPENGL_H_
