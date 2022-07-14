cmake_minimum_required(VERSION 3.16)

option(WITH_TESTS "Build Tests" ON)

set(CMAKE_CXX_STANDARD 17)

# conan is required hard, library name, version and description are configured from conan
if (EXISTS ${CMAKE_CURRENT_BINARY_DIR}/conanbuildinfo.cmake)
    include(${CMAKE_CURRENT_BINARY_DIR}/conanbuildinfo.cmake)
else ()
    message(FATAL_ERROR "The file conanbuildinfo.cmake doesn't exist, you have to run conan install first")
endif ()

project(${CONAN_PACKAGE_NAME}_PROJECT VERSION ${CONAN_PACKAGE_VERSION})

conan_basic_setup(TARGETS)

set(CMAKE_PREFIX_PATH ${CMAKE_CURRENT_BINARY_DIR})
set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_BINARY_DIR})

set(TRAACT_GUI_SRC
        src/traact_gui/main.cpp
        src/traact_gui/app/TraactGuiApp.h
        src/traact_gui/app/TraactGuiApp.cpp


        src/traact_gui/file_manager.cpp
        src/traact_gui/file_manager.h
        src/traact_gui/opengl_shader.cpp
        src/traact_gui/opengl_shader.h

        src/assets/simple-shader.vs
        src/assets/simple-shader.fs
        src/traact_gui/ImGuiUtils.h
        src/traact_gui/DataflowFile.cpp
        src/traact_gui/DataflowFile.h
        src/util/fileutil.h
        src/traact_gui/NodeEditorElements.h
        src/traact_gui/NodeEditorElements.cpp

        src/traact_gui/editor/DFGElements.h
        src/traact_gui/editor/DFGElements.cpp
        src/traact_gui/editor/SRGElements.h
        src/traact_gui/editor/SRGElements.cpp
        src/traact_gui/editor/EditorUtils.h
        src/traact_gui/editor/EditorUtils.cpp
        src/traact_gui/editor/EditorElements.h
        src/traact_gui/editor/EditorElements.cpp
        src/traact_gui/editor/PatternGraphEditor.cpp
        src/traact_gui/editor/PatternGraphEditor.h
        src/traact_gui/editor/JsonEditorSerialization.h
        src/traact_gui/editor/JsonEditorSerialization.cpp src/traact_gui/editor/DetailsEditor.cpp src/traact_gui/editor/DetailsEditor.h
        src/traact_gui/app/MainApp.cpp src/traact_gui/app/MainApp.h src/traact_gui/SelectedTraactElement.cpp src/traact_gui/SelectedTraactElement.h )

set(IMGUI_BINDING_SOURCES ${CMAKE_CURRENT_BINARY_DIR}/imgui_bindings/imgui_impl_glfw.h
        ${CMAKE_CURRENT_BINARY_DIR}/imgui_bindings/imgui_impl_glfw.cpp
        ${CMAKE_CURRENT_BINARY_DIR}/imgui_bindings/imgui_impl_opengl3.h
        ${CMAKE_CURRENT_BINARY_DIR}/imgui_bindings/imgui_impl_opengl3.cpp)

set(IMGUI_EXTERNAL_SOURCES src/external/ImFileDialog/ImFileDialog.h
        src/external/ImFileDialog/ImFileDialog.cpp
        src/external/imgui-node-editor/utilities/widgets.h
        src/external/imgui-node-editor/utilities/widgets.cpp
        src/external/imgui-node-editor/utilities/drawing.h
        src/external/imgui-node-editor/utilities/drawing.cpp
        src/external/imgui-node-editor/crude_json.cpp
        src/external/imgui-node-editor/crude_json.h
        src/external/imgui-node-editor/imgui_bezier_math.h
        src/external/imgui-node-editor/imgui_bezier_math.inl
        src/external/imgui-node-editor/imgui_canvas.cpp
        src/external/imgui-node-editor/imgui_canvas.cpp
        src/external/imgui-node-editor/imgui_canvas.h
        src/external/imgui-node-editor/imgui_canvas.h
        src/external/imgui-node-editor/imgui_extra_math.h
        src/external/imgui-node-editor/imgui_extra_math.inl
        src/external/imgui-node-editor/imgui_node_editor_api.cpp
        src/external/imgui-node-editor/imgui_node_editor_internal.h
        src/external/imgui-node-editor/imgui_node_editor_internal.inl
        src/external/imgui-node-editor/imgui_node_editor.cpp
        src/external/imgui-node-editor/imgui_node_editor.h
        src/external/imgui_misc/imgui_stdlib.h
        src/external/imgui_misc/imgui_stdlib.cpp

        src/traact_gui/debug_run/DebugRun.cpp src/traact_gui/debug_run/DebugRun.h src/traact_gui/debug_run/DebugRenderer.cpp src/traact_gui/debug_run/DebugRenderer.h src/traact_gui/debug_run/DebugRenderComponent.cpp src/traact_gui/debug_run/DebugRenderComponent.h src/traact_gui/debug_run/ImageComponent.cpp src/traact_gui/debug_run/ImageComponent.h
        src/traact_gui/debug_run/traact_opengl.h src/traact_gui/debug_run/KeyPointListComponent.cpp src/traact_gui/debug_run/KeyPointListComponent.h src/traact_gui/debug_run/Position3DComponent.cpp src/traact_gui/debug_run/Position3DComponent.h src/traact_gui/debug_run/CalibrationComponent.cpp src/traact_gui/debug_run/CalibrationComponent.h src/traact_gui/debug_run/Pose6DComponent.cpp src/traact_gui/debug_run/Pose6DComponent.h src/traact_gui/app/Window.h src/traact_gui/state/ApplicationState.cpp src/traact_gui/state/ApplicationState.h src/traact_gui/app/Window.cpp src/traact_gui/state/OpenFiles.cpp src/traact_gui/state/OpenFiles.h src/traact_gui/window/MenuFile.cpp src/traact_gui/window/MenuFile.h)

#find_package(imgui CONFIG)
#find_package(glfw3 CONFIG)
#find_package(glew CONFIG)

#add_subdirectory(external)
#add_subdirectory(src)

add_executable( ${CONAN_PACKAGE_NAME}
        ${TRAACT_GUI_SRC}
        ${IMGUI_BINDING_SOURCES}
        ${IMGUI_EXTERNAL_SOURCES})


add_custom_command(TARGET ${CONAN_PACKAGE_NAME}
        POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/src/assets/simple-shader.vs ${CMAKE_CURRENT_BINARY_DIR}/bin
        COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/src/assets/simple-shader.fs ${CMAKE_CURRENT_BINARY_DIR}/bin
        )

target_compile_definitions(${CONAN_PACKAGE_NAME} PUBLIC IMGUI_IMPL_OPENGL_LOADER_GLEW)

conan_target_link_libraries(${CONAN_PACKAGE_NAME} stdc++fs)
#target_link_libraries(${CONAN_PACKAGE_NAME} imgui::imgui GLEW::glew_s glfw::glfw)


set_target_properties(${CONAN_PACKAGE_NAME} PROPERTIES VERSION ${CONAN_PACKAGE_VERSION})
#set_target_properties(${CONAN_PACKAGE_NAME} PROPERTIES PUBLIC_HEADER "${LIB_HEADERS}")

target_include_directories(${CONAN_PACKAGE_NAME} PUBLIC
        $<INSTALL_INTERFACE:include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/imgui_bindings>
        ${CONAN_INCLUDE_DIRS_NLOHMANN_JSON}
        )
target_include_directories(${CONAN_PACKAGE_NAME} PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${CMAKE_CURRENT_BINARY_DIR}/
        )
