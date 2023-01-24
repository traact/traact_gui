cmake_minimum_required(VERSION 3.16)

option(WITH_TESTS "Build Tests" ON)

set(CMAKE_CXX_STANDARD 17)

# conan is required hard, library name, version and description are configured from conan
if (EXISTS ${CMAKE_CURRENT_BINARY_DIR}/conanbuildinfo.cmake)
    include(${CMAKE_CURRENT_BINARY_DIR}/conanbuildinfo.cmake)
else ()
    message(FATAL_ERROR "The file conanbuildinfo.cmake doesn't exist, you have to run conan install first")
endif ()

include(CheckLanguage)
check_language(CUDA)
enable_language(CUDA)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 75)
endif()

if(NOT DEFINED CMAKE_CUDA_STANDARD)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif()

project(${CONAN_PACKAGE_NAME}_PROJECT VERSION ${CONAN_PACKAGE_VERSION} LANGUAGES CUDA CXX)

conan_basic_setup(TARGETS)

set(CMAKE_PREFIX_PATH ${CMAKE_CURRENT_BINARY_DIR})
set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_BINARY_DIR})



set(IMGUI_BINDING_SOURCES ${CMAKE_CURRENT_BINARY_DIR}/imgui_bindings/imgui_impl_glfw.h
        ${CMAKE_CURRENT_BINARY_DIR}/imgui_bindings/imgui_impl_glfw.cpp
        ${CMAKE_CURRENT_BINARY_DIR}/imgui_bindings/imgui_impl_opengl3.h
        ${CMAKE_CURRENT_BINARY_DIR}/imgui_bindings/imgui_impl_opengl3.cpp)

set(IMGUI_EXTERNAL_SOURCES src/external/ImFileDialog/ImFileDialog.h
        src/external/ImFileDialog/ImFileDialog.cpp
        src/external/imgui-node-editor/utilities/builders.h
        src/external/imgui-node-editor/utilities/builders.cpp
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
        src/external/ImTerm/charconv_replacement.cpp
        src/external/ImTerm/charconv_replacement.h
        src/external/ImTerm/misc.hpp
        src/external/ImTerm/terminal.hpp
        #src/external/ImTerm/terminal.cpp
        src/external/ImTerm/terminal_commands.cpp
        src/external/ImTerm/terminal_commands.hpp
        src/external/ImTerm/terminal_helpers.hpp
        src/external/ImTerm/utils.hpp

        )

#find_package(imgui CONFIG)
#find_package(glfw3 CONFIG)
#find_package(glew CONFIG)

#add_subdirectory(external)
#add_subdirectory(src)

set(TRAACT_GUI_SRC
        src/traact/main.cpp
        src/traact/app/TraactGuiApp.h
        src/traact/app/TraactGuiApp.cpp


        src/traact/file_manager.cpp
        src/traact/file_manager.h
        src/traact/opengl_shader.cpp
        src/traact/opengl_shader.h

        src/assets/simple-shader.vs
        src/assets/simple-shader.fs
        src/traact/ImGuiUtils.h
        src/traact/DataflowFile.cpp
        src/traact/DataflowFile.h
        src/util/fileutil.h
        src/traact/NodeEditorElements.h
        src/traact/NodeEditorElements.cpp

        src/traact/editor/DFGElements.h
        src/traact/editor/DFGElements.cpp
        src/traact/editor/SRGElements.h
        src/traact/editor/SRGElements.cpp
        src/traact/editor/EditorUtils.h
        src/traact/editor/EditorUtils.cpp
        src/traact/editor/EditorElements.h
        src/traact/editor/EditorElements.cpp
        src/traact/editor/PatternGraphEditor.cpp
        src/traact/editor/PatternGraphEditor.h
        src/traact/editor/JsonEditorSerialization.h
        src/traact/editor/JsonEditorSerialization.cpp src/traact/editor/DetailsEditor.cpp src/traact/editor/DetailsEditor.h
        src/traact/app/MainApp.cpp src/traact/app/MainApp.h src/traact/SelectedTraactElement.cpp src/traact/SelectedTraactElement.h
        src/traact/debug_run/DebugRun.cpp
        src/traact/debug_run/DebugRun.h
        src/traact/debug_run/DebugRenderer.cpp
        src/traact/debug_run/DebugRenderer.h
        src/traact/debug_run/DebugRenderComponent.cpp
        src/traact/debug_run/DebugRenderComponent.h
        src/traact/debug_run/ImageComponent.cpp
        src/traact/debug_run/ImageComponent.h
        src/traact/opengl/traact_opengl.h
        src/traact/debug_run/KeyPointListComponent.cpp
        src/traact/debug_run/KeyPointListComponent.h
        src/traact/debug_run/Position3DComponent.cpp
        src/traact/debug_run/Position3DComponent.h
        src/traact/debug_run/CalibrationComponent.cpp
        src/traact/debug_run/CalibrationComponent.h
        src/traact/debug_run/Pose6DComponent.cpp
        src/traact/debug_run/Pose6DComponent.h
        src/traact/app/Window.h
        src/traact/state/ApplicationState.cpp
        src/traact/state/ApplicationState.h
        src/traact/app/Window.cpp
        src/traact/state/OpenFiles.cpp
        src/traact/state/OpenFiles.h
        src/traact/window/MenuFile.cpp
        src/traact/window/MenuFile.h
        src/traact/window/WindowOpenFiles.cpp
        src/traact/window/WindowOpenFiles.h
        src/traact/window/WindowRun.cpp
        src/traact/window/WindowRun.h
        src/traact/debug_run/PointCloudComponent.cpp
        src/traact/debug_run/PointCloudComponent.h
        src/traact/debug_run/scene/Window.cpp
        src/traact/debug_run/scene/Window.h
        src/traact/debug_run/scene/component/Camera.cpp
        src/traact/debug_run/scene/component/Camera.h
        src/traact/debug_run/scene/component/RenderPointCloud.cpp
        src/traact/debug_run/scene/component/RenderPointCloud.h
        src/traact/debug_run/scene/Component.cpp
        src/traact/debug_run/scene/Component.h
        src/traact/debug_run/scene/Object.cpp
        src/traact/debug_run/scene/Object.h
        src/traact/debug_run/scene/Transform.cpp
        src/traact/debug_run/scene/Transform.h
        src/traact/debug_run/DebugSceneWindow.cpp
        src/traact/debug_run/DebugSceneWindow.h
        src/traact/window/WindowDetails.cpp src/traact/window/WindowDetails.h src/traact/window/WindowDataflow.cpp src/traact/window/WindowDataflow.h src/traact/debug_run/scene/component/RenderCoordinateSystem.cpp src/traact/debug_run/scene/component/RenderCoordinateSystem.h src/traact/debug_run/scene/component/PoseSource.cpp src/traact/debug_run/scene/component/PoseSource.h src/traact/application_data/ApplicationData.cpp src/traact/application_data/ApplicationData.h src/traact/application_data/source/RawSource.cpp src/traact/application_data/source/RawSource.h src/traact/application_data/DataPort.cpp src/traact/application_data/DataPort.h src/traact/application_data/application_data.h src/traact/application_data/source/OpenGlTextureSource.cpp src/traact/application_data/source/OpenGlTextureSource.h src/traact/debug_run/scene/component/IdentityRotation.cpp src/traact/debug_run/scene/component/IdentityRotation.h src/traact/debug_run/scene/component/FileReaderWriter.cpp src/traact/debug_run/scene/component/FileReaderWriter.h src/traact/debug_run/scene/component/StaticPose.cpp src/traact/debug_run/scene/component/StaticPose.h src/traact/debug_run/magnum/Window.cpp src/traact/debug_run/magnum/Window.h src/traact/opengl/Framebuffer.cpp src/traact/opengl/Framebuffer.h src/traact/debug_run/magnum/feature/DrawTriangle.cpp src/traact/debug_run/magnum/feature/DrawTriangle.h src/traact/debug_run/magnum/magnum_definitions.h src/traact/state/project/Graph.cpp src/traact/state/project/Graph.h src/traact/state/project/pattern/Pattern.cpp src/traact/state/project/pattern/Pattern.h src/traact/state/project/Project.cpp src/traact/state/project/Project.h src/traact/state/project/graph/Node.cpp src/traact/state/project/graph/Node.h src/traact/state/project/graph/Edge.cpp src/traact/state/project/graph/Edge.h src/traact/state/project/pattern/CompositePattern.cpp src/traact/state/project/pattern/CompositePattern.h src/traact/state/project/pattern/PatternInstance.cpp src/traact/state/project/pattern/PatternInstance.h src/traact/window/OpenProjects.cpp src/traact/window/OpenProjects.h src/traact/state/project/pattern/FacadePatternFactory.cpp src/traact/state/project/pattern/FacadePatternFactory.h src/traact/state/project/pattern/PatternFactory.h src/traact/state/project/pattern/GlobalPatternFactory.cpp src/traact/state/project/pattern/GlobalPatternFactory.h src/traact/state/project/pattern/PatternInfo.cpp src/traact/state/project/pattern/PatternInfo.h src/traact/imgui_util.h src/traact/imgui_gl_util.cpp src/traact/imgui_gl_util.h src/external/ImTerm/charconv_replacement.h src/external/ImTerm/charconv_replacement.cpp src/traact/window/WindowLog.cpp src/traact/window/WindowLog.h src/traact/window/WindowPatterns.cpp src/traact/window/WindowPatterns.h src/traact/debug_run/DataflowProfileWindow.cpp src/traact/debug_run/DataflowProfileWindow.h)

add_executable( ${CONAN_PACKAGE_NAME}
        ${TRAACT_GUI_SRC}
        ${IMGUI_BINDING_SOURCES}
        ${IMGUI_EXTERNAL_SOURCES} )

include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

add_custom_command(TARGET ${CONAN_PACKAGE_NAME}
        POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/src/assets/simple-shader.vs ${CMAKE_CURRENT_BINARY_DIR}/bin
        COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/src/assets/simple-shader.fs ${CMAKE_CURRENT_BINARY_DIR}/bin
        )

target_compile_definitions(${CONAN_PACKAGE_NAME} PUBLIC IMGUI_IMPL_OPENGL_LOADER_GLEW)

#conan_target_link_libraries(${CONAN_PACKAGE_NAME} stdc++fs)
#target_link_libraries(${CONAN_PACKAGE_NAME} imgui::imgui GLEW::glew_s glfw::glfw)

find_package(CUDAToolkit REQUIRED)

target_link_libraries(${CONAN_PACKAGE_NAME} stdc++fs)
target_link_libraries(${CONAN_PACKAGE_NAME} CONAN_PKG::spdlog)
target_link_libraries(${CONAN_PACKAGE_NAME} CONAN_PKG::traact_core)
target_link_libraries(${CONAN_PACKAGE_NAME} CONAN_PKG::traact_spatial)
target_link_libraries(${CONAN_PACKAGE_NAME} CONAN_PKG::traact_vision)
target_link_libraries(${CONAN_PACKAGE_NAME} CONAN_PKG::traact_component_pointcloud)
target_link_libraries(${CONAN_PACKAGE_NAME} CONAN_PKG::opencv)
target_link_libraries(${CONAN_PACKAGE_NAME} CONAN_PKG::glfw)

target_link_libraries(${CONAN_PACKAGE_NAME} CONAN_PKG::glew)
target_link_libraries(${CONAN_PACKAGE_NAME} CONAN_PKG::ghc-filesystem)
target_link_libraries(${CONAN_PACKAGE_NAME} CONAN_PKG::imgui)
target_link_libraries(${CONAN_PACKAGE_NAME} CONAN_PKG::imguizmo)
target_link_libraries(${CONAN_PACKAGE_NAME} CONAN_PKG::implot)
target_link_libraries(${CONAN_PACKAGE_NAME} CONAN_PKG::stb)
target_link_libraries(${CONAN_PACKAGE_NAME} CONAN_PKG::nodesoup)
target_link_libraries(${CONAN_PACKAGE_NAME} CONAN_PKG::open3d)
target_link_libraries(${CONAN_PACKAGE_NAME} CONAN_PKG::glm)

target_link_libraries(${CONAN_PACKAGE_NAME} CONAN_PKG::zlib)
target_link_libraries(${CONAN_PACKAGE_NAME} CONAN_PKG::corrade)
target_link_libraries(${CONAN_PACKAGE_NAME} CONAN_PKG::magnum)
target_link_libraries(${CONAN_PACKAGE_NAME} CONAN_PKG::magnum-integration)
target_link_libraries(${CONAN_PACKAGE_NAME} CONAN_PKG::magnum-plugins)
target_link_libraries(${CONAN_PACKAGE_NAME} CONAN_PKG::re2)

target_link_libraries(${CONAN_PACKAGE_NAME} CUDA::cudart)


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
