/** Copyright (C) 2022  Frieder Pankratz <frieder.pankratz@gmail.com> **/
#include "MainApp.h"
#include "TraactGuiApp.h"

static void GlfwErrorCallback(int error, const char *description) {
    SPDLOG_ERROR("Glfw Error {0}: {1}\n", error, description);
}

namespace traact::gui {

void MainApp::init() {
    initOpenGl();
    initImGui();
    initFileDialog();
}
void MainApp::initImGui() const {
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();

    ImGuiIO &gui_io = ImGui::GetIO();
    gui_io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;
    gui_io.ConfigFlags |= ImGuiConfigFlags_ViewportsEnable;
    gui_io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;



    // Setup Dear ImGui style
    ImGui::StyleColorsDark();

    // Setup Platform/Renderer bindings
    if (!ImGui_ImplGlfw_InitForOpenGL(window_, true)) {
        throw std::runtime_error("ImGui_ImplGlfw_InitForOpenGL failed");
    }

    if (!ImGui_ImplOpenGL3_Init(glsl_version_.c_str())) {
        throw std::runtime_error("ImGui_ImplOpenGL3_Init failed");
    }

}
void MainApp::initOpenGl() {
    glfwSetErrorCallback(GlfwErrorCallback);
    if (glfwInit() == GLFW_FALSE) {
        throw std::runtime_error("glfwInit failed");
    }

    // Decide GL+GLSL versions
#if __APPLE__
    // GL 3.2 + GLSL 150
        glsl_version_ = "#version 150";
        glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
        glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 2);
        glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE); // 3.2+ only
        glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);		   // Required on Mac
#else
// GL 3.0 + GLSL 130
    glsl_version_ = "#version 130";
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);
#endif

    // Create window with graphics context

    window_ = glfwCreateWindow(1280, 720, "Traact GUI", NULL, NULL);
    if (window_ == NULL) {
        throw std::runtime_error("could not create glfw window");
    }

    glfwMakeContextCurrent(window_);
    glfwSwapInterval(1); // Enable vsync


    if (glewInit() != GLEW_OK) {
        throw std::runtime_error("Failed to initialize OpenGL loader!");
    }

    glfwGetFramebufferSize(window_, &screen_width_, &screen_height_);
    glViewport(0, 0, screen_width_, screen_height_);
}
void MainApp::blockingLoop() {
    ImGuiIO &gui_io = ImGui::GetIO();
    while (running_) {
        glfwPollEvents();

        // feed inputs to dear imgui, start new frame
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();



        // We demonstrate using the full viewport area or the work area (without menu-bars, task-bars etc.)
        // Based on your use case you may want one of the other.
        const ImGuiViewport *viewport = ImGui::GetMainViewport();
        ImGui::SetNextWindowPos(viewport->WorkPos);
        ImGui::SetNextWindowSize(viewport->WorkSize);

        ImGui::DockSpaceOverViewport(ImGui::GetMainViewport());

        if (should_stop_) {
            running_ = traact_app_.onFrameStop();
        } else {
            traact_app_.onFrame();
        }


        ImGui::Render();


        int display_w, display_h;
        glfwGetFramebufferSize(window_, &display_w, &display_h);
        glViewport(0, 0, display_w, display_h);
        glClearColor(0.45f, 0.55f, 0.60f, 1.00f);
        glClear(GL_COLOR_BUFFER_BIT);

        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

        if(gui_io.ConfigFlags & ImGuiConfigFlags_ViewportsEnable){
            GLFWwindow *backup_current_context = glfwGetCurrentContext();
            ImGui::UpdatePlatformWindows();
            ImGui::RenderPlatformWindowsDefault();
            glfwMakeContextCurrent(backup_current_context);
        }

        glfwSwapBuffers(window_);

        if (glfwWindowShouldClose(window_)) {
            should_stop_ = true;
        }
    }
}
void MainApp::dispose() const {
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();

    glfwDestroyWindow(window_);
    glfwTerminate();
}
void MainApp::initFileDialog() const {
    ifd::FileDialog::Instance().CreateTexture = [](uint8_t *data, int w, int h, char fmt) -> void * {
        GLuint tex;

        glGenTextures(1, &tex);
        glBindTexture(GL_TEXTURE_2D, tex);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, w, h, 0, (fmt == 0) ? GL_BGRA : GL_RGBA, GL_UNSIGNED_BYTE, data);
        glGenerateMipmap(GL_TEXTURE_2D);
        glBindTexture(GL_TEXTURE_2D, 0);

        //return (void*)tex;
        return (void *) ((uintptr_t) tex);
    };
    ifd::FileDialog::Instance().DeleteTexture = [](void *tex) {
        GLuint texID = (GLuint) ((uintptr_t) tex);
        glDeleteTextures(1, &texID);
    };
}
MainApp::MainApp(std::atomic_bool &should_stop) : should_stop_(should_stop) {
    init();
}
void MainApp::loadDataflow(std::string dataflow_file) {
    traact_app_.openFile(dataflow_file);
}
MainApp::~MainApp() {
    dispose();
}
}