/** Copyright (C) 2022  Frieder Pankratz <frieder.pankratz@gmail.com> **/

#ifndef TRAACT_GUI_SRC_TRAACT_GUI_DEBUG_RUN_SCENE_COMPONENT_FILEREADERWRITER_H_
#define TRAACT_GUI_SRC_TRAACT_GUI_DEBUG_RUN_SCENE_COMPONENT_FILEREADERWRITER_H_

#include "traact_gui/debug_run/scene/Component.h"
#include <traact/component/generic/FileReaderWriter.h>

namespace traact::gui::scene::component {

class FileReaderWriter : public Component{
 public:
    FileReaderWriter(const std::shared_ptr<Object> &object, const std::string &name);
    void setReader(const std::shared_ptr<traact::component::FileReaderWriterRead<spatial::Pose6DHeader>> &reader);
    void setWriter(const std::shared_ptr<traact::component::FileReaderWriterWrite<spatial::Pose6DHeader>> &writer);
    virtual void update() override;
    virtual void drawGui() override;
 private:
    enum class Mode {
        Use,
        Modify
    };
    std::shared_ptr<traact::component::FileReaderWriterRead<spatial::Pose6DHeader>> reader_;
    std::shared_ptr<traact::component::FileReaderWriterWrite<spatial::Pose6DHeader>> writer_;
    Mode current_mode_{Mode::Use};

    void useCalibration();
    void saveCalibration();
};

} // traact

#endif //TRAACT_GUI_SRC_TRAACT_GUI_DEBUG_RUN_SCENE_COMPONENT_FILEREADERWRITER_H_
