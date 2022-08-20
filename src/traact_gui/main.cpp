#include "traact_gui/app/MainApp.h"
#include <spdlog/spdlog.h>
#include <signal.h>
#include <opencv2/core/utility.hpp>
#include <traact/util/Logging.h>
std::atomic_bool should_stop{false};

void ctrlC(int i) {
    SPDLOG_INFO("User requested exit (Ctrl-C).");
    should_stop = true;
}

void runApp(std::optional<std::string> &dataflow_file);

int main(int argc, char** argv)
{
    using namespace traact::gui;

    traact::util::initLogging(spdlog::level::level_enum::info);

    signal(SIGINT, ctrlC);

    const std::string keys =
        "{help h usage ? |      | print this message   }"
        "{@dataflow      |<none>| load traact dataflow }";

    cv::CommandLineParser parser(argc, argv, keys);
    parser.about("Traact GUI v0.0.1");
    if (parser.has("help")) {
        parser.printMessage();
        return 0;
    }
    if (!parser.check()) {
        parser.printErrors();
        return 0;
    }

    std::optional<std::string> dataflow_file{};
    if (parser.has("@dataflow")) {
        dataflow_file = parser.get<std::string>(0);
    }

    runApp(dataflow_file);

    return 0;
}

void runApp(std::optional<std::string> &dataflow_file) {
    try{
        traact::gui::MainApp app(should_stop);

        if(dataflow_file.has_value()){
            app.loadDataflow(dataflow_file.value());
        }

        app.blockingLoop();

    }catch (std::exception& exception){
        SPDLOG_ERROR(exception.what());
    }
}

//#include <Magnum/Math/Color.h>
//#include <Magnum/GL/DefaultFramebuffer.h>
//#include <Magnum/GL/Renderer.h>
//#include <Magnum/ImGuiIntegration/Context.hpp>
//
//#ifdef CORRADE_TARGET_ANDROID
//#include <Magnum/Platform/AndroidApplication.h>
//#elif defined(CORRADE_TARGET_EMSCRIPTEN)
//#include <Magnum/Platform/EmscriptenApplication.h>
//#else
//#include <Magnum/Platform/GlfwApplication.h>
//#endif
//#include <Magnum/SceneGraph/Scene.h>
//#include <Magnum/SceneGraph/MatrixTransformation3D.h>
//#include <Magnum/SceneGraph/Camera.h>
//#include <Magnum/SceneGraph/Drawable.h>
//#include <spdlog/spdlog.h>
//
//namespace Magnum { namespace Examples {
//
//typedef SceneGraph::Object<SceneGraph::MatrixTransformation3D> Object3D;
//typedef SceneGraph::Scene<SceneGraph::MatrixTransformation3D> Scene3D;
//
//using namespace Math::Literals;
//
//class ImGuiExample: public Platform::Application {
// public:
//    explicit ImGuiExample(const Arguments& arguments);
//
//    void drawEvent() override;
//
//    void viewportEvent(ViewportEvent& event) override;
//
//    void keyPressEvent(KeyEvent& event) override;
//    void keyReleaseEvent(KeyEvent& event) override;
//
//    void mousePressEvent(MouseEvent& event) override;
//    void mouseReleaseEvent(MouseEvent& event) override;
//    void mouseMoveEvent(MouseMoveEvent& event) override;
//    void mouseScrollEvent(MouseScrollEvent& event) override;
//    void textInputEvent(TextInputEvent& event) override;
//
// private:
//    ImGuiIntegration::Context _imgui{NoCreate};
//
//    bool _showDemoWindow = true;
//    bool _showAnotherWindow = false;
//    Color4 _clearColor = 0x72909aff_rgbaf;
//    Float _floatValue = 0.0f;
//
//    Scene3D _scene;
//    Object3D* _cameraObject;
//    SceneGraph::Camera3D* _camera;
//    SceneGraph::DrawableGroup3D _drawables;
//};
//
//ImGuiExample::ImGuiExample(const Arguments& arguments): Platform::Application{arguments,
//                                                                              Configuration{}.setTitle("Magnum ImGui Example")
//                                                                                  .setWindowFlags(Configuration::WindowFlag::Resizable)}
//{
//    _imgui = ImGuiIntegration::Context(Vector2{windowSize()}/dpiScaling(),
//                                       windowSize(), framebufferSize());
//
//    // strange bug, caused by old check of keymap, check could be disabled with a #DEFINE IMGUI_DISABLE_OBSOLETE_KEYIO for imgui.cpp
//    auto& io = ImGui::GetIO();
//    for(int i=0;i<645;++i){
//        if(io.KeyMap[i]>511){
//            io.KeyMap[i] = -1;
//        }
//
//    }
//
//    ImGuiIO &gui_io = ImGui::GetIO();
//    gui_io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;
//    gui_io.ConfigFlags |= ImGuiConfigFlags_ViewportsEnable;
//    gui_io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
//    // Setup Dear ImGui style
//    ImGui::StyleColorsDark();
//    auto& style = ImGui::GetStyle();
//    style.WindowMinSize = ImVec2(320,240);
//
//
//    /* Set up proper blending to be used by ImGui. There's a great chance
//       you'll need this exact behavior for the rest of your scene. If not, set
//       this only for the drawFrame() call. */
//    GL::Renderer::setBlendEquation(GL::Renderer::BlendEquation::Add,
//                                   GL::Renderer::BlendEquation::Add);
//    GL::Renderer::setBlendFunction(GL::Renderer::BlendFunction::SourceAlpha,
//                                   GL::Renderer::BlendFunction::OneMinusSourceAlpha);
//
//    /* Configure camera */
//    _cameraObject = new Object3D{&_scene};
//    _cameraObject->translate(Vector3::zAxis(5.0f));
//    _camera = new SceneGraph::Camera3D{*_cameraObject};
//    _camera->setAspectRatioPolicy(SceneGraph::AspectRatioPolicy::Extend)
//        .setProjectionMatrix(Matrix4::perspectiveProjection(35.0_degf, 4.0f/3.0f, 0.001f, 100.0f))
//        .setViewport(GL::defaultFramebuffer.viewport().size());
//
//#if !defined(MAGNUM_TARGET_WEBGL) && !defined(CORRADE_TARGET_ANDROID)
//    /* Have some sane speed, please */
//    //setMinimalLoopPeriod(16);
//#endif
//}
//
//void ImGuiExample::drawEvent() {
//    GL::defaultFramebuffer.clear(GL::FramebufferClear::Color);
//
//    _imgui.newFrame();
//
//    const ImGuiViewport *viewport = ImGui::GetMainViewport();
//    ImGui::SetNextWindowPos(viewport->WorkPos);
//    ImGui::SetNextWindowSize(viewport->WorkSize);
//
//    ImGui::DockSpaceOverViewport(ImGui::GetMainViewport());
//
//    /* Enable text input, if needed */
//    if(ImGui::GetIO().WantTextInput && !isTextInputActive())
//        startTextInput();
//    else if(!ImGui::GetIO().WantTextInput && isTextInputActive())
//        stopTextInput();
//
//    /* 1. Show a simple window.
//       Tip: if we don't call ImGui::Begin()/ImGui::End() the widgets appear in
//       a window called "Debug" automatically */
//    {
//        ImGui::Text("Hello, world!");
//        ImGui::SliderFloat("Float", &_floatValue, 0.0f, 1.0f);
//        if(ImGui::ColorEdit3("Clear Color", _clearColor.data()))
//            GL::Renderer::setClearColor(_clearColor);
//        if(ImGui::Button("Test Window"))
//            _showDemoWindow ^= true;
//        if(ImGui::Button("Another Window"))
//            _showAnotherWindow ^= true;
//        ImGui::Text("Application average %.3f ms/frame (%.1f FPS)",
//                    1000.0/Double(ImGui::GetIO().Framerate), Double(ImGui::GetIO().Framerate));
//    }
//
//    /* 2. Show another simple window, now using an explicit Begin/End pair */
//    if(_showAnotherWindow) {
//        ImGui::SetNextWindowSize(ImVec2(500, 100), ImGuiCond_FirstUseEver);
//        ImGui::Begin("Another Window", &_showAnotherWindow);
//        ImGui::Text("Hello");
//        ImGui::End();
//    }
//
//    /* 3. Show the ImGui demo window. Most of the sample code is in
//       ImGui::ShowDemoWindow() */
//    if(_showDemoWindow) {
//        ImGui::SetNextWindowPos(ImVec2(650, 20), ImGuiCond_FirstUseEver);
//        ImGui::ShowDemoWindow();
//    }
//
//    /* Update application cursor */
//    _imgui.updateApplicationCursor(*this);
//
//    /* Set appropriate states. If you only draw ImGui, it is sufficient to
//       just enable blending and scissor test in the constructor. */
//    GL::Renderer::enable(GL::Renderer::Feature::Blending);
//    GL::Renderer::enable(GL::Renderer::Feature::ScissorTest);
//    GL::Renderer::disable(GL::Renderer::Feature::FaceCulling);
//    GL::Renderer::disable(GL::Renderer::Feature::DepthTest);
//
//    _imgui.drawFrame();
//
//    /* Reset state. Only needed if you want to draw something else with
//       different state after. */
//    GL::Renderer::enable(GL::Renderer::Feature::DepthTest);
//    GL::Renderer::enable(GL::Renderer::Feature::FaceCulling);
//    GL::Renderer::disable(GL::Renderer::Feature::ScissorTest);
//    GL::Renderer::disable(GL::Renderer::Feature::Blending);
//
//    _camera->draw(_drawables);
//
//    swapBuffers();
//    redraw();
//}
//
//void ImGuiExample::viewportEvent(ViewportEvent& event) {
//    GL::defaultFramebuffer.setViewport({{}, event.framebufferSize()});
//
//    _imgui.relayout(Vector2{event.windowSize()}/event.dpiScaling(),
//                    event.windowSize(), event.framebufferSize());
//}
//
//void ImGuiExample::keyPressEvent(KeyEvent& event) {
//    if(_imgui.handleKeyPressEvent(event)) return;
//}
//
//void ImGuiExample::keyReleaseEvent(KeyEvent& event) {
//    if(_imgui.handleKeyReleaseEvent(event)) return;
//}
//
//void ImGuiExample::mousePressEvent(MouseEvent& event) {
//    if(_imgui.handleMousePressEvent(event)) return;
//}
//
//void ImGuiExample::mouseReleaseEvent(MouseEvent& event) {
//    if(_imgui.handleMouseReleaseEvent(event)) return;
//}
//
//void ImGuiExample::mouseMoveEvent(MouseMoveEvent& event) {
//    if(_imgui.handleMouseMoveEvent(event)) return;
//}
//
//void ImGuiExample::mouseScrollEvent(MouseScrollEvent& event) {
//    if(_imgui.handleMouseScrollEvent(event)) {
//        /* Prevent scrolling the page */
//        event.setAccepted();
//        return;
//    }
//}
//
//void ImGuiExample::textInputEvent(TextInputEvent& event) {
//    if(_imgui.handleTextInputEvent(event)) return;
//}
//
//}}
//
//MAGNUM_APPLICATION_MAIN(Magnum::Examples::ImGuiExample)