/** Copyright (C) 2022  Frieder Pankratz <frieder.pankratz@gmail.com> **/

#ifndef TRAACT_GUI_SRC_TRAACT_GUI_DEBUG_RUN_IMAGECOMPONENT_H_
#define TRAACT_GUI_SRC_TRAACT_GUI_DEBUG_RUN_IMAGECOMPONENT_H_

#include "DebugRenderComponent.h"
#include "traact_opengl.h"

namespace traact::gui {


class ImageComponent : public DebugRenderComponent{
 public:
    ImageComponent(int port_index,
                   const std::string &port_name,
                   const std::string &window_name,
                   DebugRenderer *renderer);
    virtual ~ImageComponent() = default;

    void update(buffer::ComponentBuffer &buffer, std::vector<RenderCommand> &additional_commands) override;

 private:
    GLuint texture_;
    bool init_texture_{false};
    double fps_{0};
    TimestampSteady last_ts_{TimestampSteady::min()};
    int valid_count_{0};
    int upload_count_{0};
    Timestamp last_image_upload_{kTimestampZero};
    GLuint pbo_id_;
    void uploadImage(traact::buffer::ComponentBuffer &data);
    void draw();

};

} // traact

#endif //TRAACT_GUI_SRC_TRAACT_GUI_DEBUG_RUN_IMAGECOMPONENT_H_
