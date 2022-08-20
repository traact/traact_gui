/** Copyright (C) 2022  Frieder Pankratz <frieder.pankratz@gmail.com> **/

#ifndef TRAACT_GUI_SRC_TRAACT_GUI_DEBUG_RUN_SCENE_OBJECT_H_
#define TRAACT_GUI_SRC_TRAACT_GUI_DEBUG_RUN_SCENE_OBJECT_H_

#include "Component.h"
#include "Transform.h"

namespace traact::gui::scene {

 class Window;

 class Object : public std::enable_shared_from_this<Object>{
 public:
    using SharedPtr = std::shared_ptr<Object>;

    Object(Window* window);
    ~Object() = default;

    void init();
    void update(buffer::ComponentBuffer &buffer, std::vector<RenderCommand> &additional_commands);
    void draw();
    void drawGui();
    void stop();

    const Transform::SharedPtr& getTransform();

    template<class T> std::shared_ptr<T> getComponent(const std::string& name){
        auto component = components_.find(name);
        if(component != components_.end()){
            return std::dynamic_pointer_cast<T>(component->second);
        } else {
            auto new_component = std::make_shared<T>(shared_from_this(), name);
            components_.template emplace(name, new_component);
            return new_component;
        }

    }
     template<class T> std::shared_ptr<T> addComponent(const std::string& name){
         auto component = components_.find(name);
         if(component != components_.end()){
             throw std::invalid_argument(fmt::format("component already exists: {0}", name));
         } else {
             auto new_component = std::make_shared<T>(shared_from_this(), name);
             components_.template emplace(name, new_component);
             return new_component;
         }

     }
    void addComponent(const std::string& name, Component::SharedPtr component);

    std::shared_ptr<component::Camera> getMainCamera() const;

 private:
    Transform::SharedPtr transform_;
     Window* window_;
     std::map<std::string,Component::SharedPtr> components_;
};

} // traact

#endif //TRAACT_GUI_SRC_TRAACT_GUI_DEBUG_RUN_SCENE_OBJECT_H_
