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

static const constexpr GLint getOpenGlInternalLuminanceFormat(BaseType data_type) {
    switch (data_type) {
        case INT_8: return GL_LUMINANCE8;
        case UINT_8:return GL_LUMINANCE8UI_EXT;
        case INT_16:return GL_LUMINANCE16I_EXT;
        case UINT_16:return GL_LUMINANCE16UI_EXT;
        case INT_32:return GL_LUMINANCE32I_EXT;
        case FLOAT_16:return GL_LUMINANCE16F_ARB;
        case FLOAT_32:return GL_LUMINANCE32F_ARB;
        case FLOAT_64:
        case UNKNOWN: return GL_INVALID_ENUM;
    }
}

static const constexpr GLint getOpenGlInternalRgbFormat(BaseType data_type) {
    switch (data_type) {
        case INT_8: return GL_RGB8I;
        case UINT_8:return GL_RGB8UI;
        case INT_16:return GL_RGB16I;
        case UINT_16:return GL_RGB16UI;
        case INT_32:return GL_RGB32I;
        case FLOAT_16:return GL_RGB16F;
        case FLOAT_32:return GL_RGB32F;
        case FLOAT_64:
        case UNKNOWN: return GL_INVALID_ENUM;
    }
}

static const constexpr GLint getOpenGlInternalRgbaFormat(BaseType data_type) {
    switch (data_type) {
        case INT_8: return GL_RGBA8I;
        case UINT_8:return GL_RGBA8UI;
        case INT_16:return GL_RGBA16I;
        case UINT_16:return GL_RGBA16UI;
        case INT_32:return GL_RGBA32I;
        case FLOAT_16:return GL_RGBA16F;
        case FLOAT_32:return GL_RGBA32F;
        case FLOAT_64:
        case UNKNOWN: return GL_INVALID_ENUM;
    }
}

static const constexpr GLint getOpenGlInternalRFormat(BaseType data_type) {
    switch (data_type) {
        case INT_8: return GL_R8;
        case UINT_8:return GL_R8UI;
        case INT_16:return GL_R16I;
        case UINT_16:return GL_R16UI;
        case INT_32:return GL_R32I;
        case FLOAT_16:return GL_R16F;
        case FLOAT_32:return GL_R32F;
        case FLOAT_64:
        case UNKNOWN: return GL_INVALID_ENUM;
    }
}
static const constexpr GLint getOpenGlInternalFloatFormat(BaseType data_type, int channels) {
    switch (data_type) {
        default: return GL_INVALID_ENUM;
        case FLOAT_32:break;
    }

    switch (channels) {
        case 1: return GL_R32F;
        case 2: return GL_RG32F;
        case 3: return GL_RGB32F;
        case 4: return GL_RGBA32F;
        default: return GL_INVALID_ENUM;

    }
}

static const constexpr GLint getOpenGlInternalFormat(BaseType data_type, vision::PixelFormat pixel_format, int channels) {
    switch (pixel_format) {
        case vision::PixelFormat::LUMINANCE:return getOpenGlInternalLuminanceFormat(data_type);
        case vision::PixelFormat::RGB: return getOpenGlInternalRgbFormat(data_type);
        case vision::PixelFormat::BGR: return getOpenGlInternalRgbFormat(data_type);
        case vision::PixelFormat::RGBA: return getOpenGlInternalRgbaFormat(data_type);
        case vision::PixelFormat::BGRA:return getOpenGlInternalRgbaFormat(data_type);
        case vision::PixelFormat::DEPTH:return getOpenGlInternalRFormat(data_type);
        case vision::PixelFormat::FLOAT:return getOpenGlInternalFloatFormat(data_type,channels);

        case vision::PixelFormat::YUV422:
        case vision::PixelFormat::YUV411:
        case vision::PixelFormat::RAW:
        case vision::PixelFormat::MJPEG:
        case vision::PixelFormat::UNKNOWN_PIXELFORMAT:return GL_INVALID_ENUM;
    }
}

static const constexpr GLint getOpenGlLuminanceFormat(BaseType data_type) {
    switch (data_type) {
        case INT_8:
        case UINT_8:
        case INT_16:
        case UINT_16:
        case INT_32:return GL_LUMINANCE_INTEGER_EXT;
        case FLOAT_16:
        case FLOAT_32:return GL_LUMINANCE;
        case FLOAT_64:
        case UNKNOWN: return GL_INVALID_ENUM;
    }
}

static const constexpr GLint getOpenGlRgbFormat(BaseType data_type) {
    switch (data_type) {
        case INT_8:
        case UINT_8:
        case INT_16:
        case UINT_16:
        case INT_32:return GL_RGB_INTEGER;
        case FLOAT_16:
        case FLOAT_32:return GL_RGB;
        case FLOAT_64:
        case UNKNOWN: return GL_INVALID_ENUM;
    }
}

static const constexpr GLint getOpenGlRgbaFormat(BaseType data_type) {
    switch (data_type) {
        case INT_8:
        case UINT_8:
        case INT_16:
        case UINT_16:
        case INT_32:return GL_RGBA_INTEGER;
        case FLOAT_16:
        case FLOAT_32:return GL_RGBA;
        case FLOAT_64:
        case UNKNOWN: return GL_INVALID_ENUM;
    }
}
static const constexpr GLint getOpenGlRFormat(BaseType data_type) {
    switch (data_type) {
        case INT_8:
        case UINT_8:
        case INT_16:
        case UINT_16:
        case INT_32:return GL_RED_INTEGER;
        case FLOAT_16:
        case FLOAT_32:return GL_R;
        case FLOAT_64:
        case UNKNOWN: return GL_INVALID_ENUM;
    }
}

static const constexpr GLint getOpenGlFloatFormat(BaseType data_type, int channels) {
    switch (data_type) {
        default: return GL_INVALID_ENUM;
        case FLOAT_32:break;
    }

    switch (channels) {
        case 1: return GL_R;
        case 2: return GL_RG;
        case 3: return GL_RGB;
        case 4: return GL_RGBA;
        default: return GL_INVALID_ENUM;

    }
}

static const constexpr GLint getOpenGlFormat(BaseType data_type, vision::PixelFormat pixel_format, int channels) {
    switch (pixel_format) {
        case vision::PixelFormat::LUMINANCE:return getOpenGlLuminanceFormat(data_type);
        case vision::PixelFormat::RGB: return getOpenGlRgbFormat(data_type);
        case vision::PixelFormat::BGR: return getOpenGlRgbFormat(data_type);
        case vision::PixelFormat::RGBA: return getOpenGlRgbaFormat(data_type);
        case vision::PixelFormat::BGRA:return getOpenGlRgbaFormat(data_type);
        case vision::PixelFormat::DEPTH:return getOpenGlRFormat(data_type);
        case vision::PixelFormat::FLOAT:return getOpenGlFloatFormat(data_type,channels);

        case vision::PixelFormat::YUV422:
        case vision::PixelFormat::YUV411:
        case vision::PixelFormat::RAW:
        case vision::PixelFormat::MJPEG:
        case vision::PixelFormat::UNKNOWN_PIXELFORMAT:return GL_INVALID_ENUM;
    }
}

static const constexpr GLint getOpenGlInternalFormat(const vision::ImageHeader& header) {
    return getOpenGlInternalFormat(header.base_type, header.pixel_format, header.channels);
}
static const constexpr GLint getOpenGlFormat(const vision::ImageHeader& header) {
    return getOpenGlFormat(header.base_type, header.pixel_format, header.channels);
}
static const constexpr GLint getOpenGlInternalFormat(const vision::GpuImageHeader& header) {
    return getOpenGlInternalFormat(header.base_type, header.pixel_format, header.channels);
}
static const constexpr GLint getOpenGlFormat(const vision::GpuImageHeader& header) {
    return getOpenGlFormat(header.base_type, header.pixel_format, header.channels);
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
    glVertex3f(-0.0 * scale, 0.0f, 0.0f);
    glVertex3f(1.0 * scale, 0.0f, 0.0f);

    glVertex3f(1.0 * scale, 0.0f, 0.0f);
    glVertex3f(0.75f * scale, 0.25f * scale, 0.0f);

    glVertex3f(1.0 * scale, 0.0f, 0.0f);
    glVertex3f(0.75f * scale, -0.25f * scale, 0.0f);
    glEnd();

    // y
    glColor3f(0.0, 1.0, 0.0);
    glBegin(GL_LINES);
    glVertex3f(0.0, -0.0f * scale, 0.0f);
    glVertex3f(0.0, 1.0f * scale, 0.0f);

    glVertex3f(0.0, 1.0f * scale, 0.0f);
    glVertex3f(0.25f * scale, 0.75f * scale, 0.0f);

    glVertex3f(0.0, 1.0f * scale, 0.0f);
    glVertex3f(-0.25f * scale, 0.75f * scale, 0.0f);
    glEnd();

    // z
    glColor3f(0.0, 0.0, 1.0);
    glBegin(GL_LINES);
    glVertex3f(0.0, 0.0f, -0.0f * scale);
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
