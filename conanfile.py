# /usr/bin/python3
import os
from conans import ConanFile, CMake, tools


class Traact(ConanFile):
    name = "traact_gui"
    version = "0.0.1"
    
    description = "GUI Editor for Traact"
    url = "https://github.com/traact/traact_gui.git"
    license = "MIT"
    author = "Frieder Pankratz"

    short_paths = True

    generators = "cmake", "TraactVirtualRunEnvGenerator"
    settings = "os", "compiler", "build_type", "arch"
    compiler = "cppstd"
    keep_imports = True
    options = {
        "shared": [True, False],
    }

    default_options = {
        "shared": True,        
    }

    exports_sources = "CMakeLists.txt", "src/*", "external/*",

    def requirements(self):
        self.requires("traact_run_env/1.0.0@traact/latest")
        self.requires("traact_core/1.0.0@traact/latest")
        self.requires("traact_spatial/1.0.0@traact/latest")
        self.requires("traact_vision/1.0.0@traact/latest")
        self.requires("traact_component_basic/1.0.0@traact/latest")
        self.requires("traact_component_kinect_azure/1.0.0@traact/latest")
        self.requires("traact_component_cereal/1.0.0@traact/latest")
        self.requires("traact_component_aruco/1.0.0@traact/latest")
        self.requires("traact_component_pcpd_shm/1.0.0@traact/latest")
        self.requires("traact_component_http/1.0.0@traact/latest")
        self.requires("opencv/4.5.5@camposs/stable")
        self.requires("glfw/3.3.4")
        self.requires("glew/2.2.0")
        self.requires("imgui/cci.20220207+1.87.docking")
        self.requires("ghc-filesystem/1.5.8")
        #self.requires("imguizmo/cci.20210720")
        self.requires("imguizmo/cci.20210223")
        self.requires("implot/0.13")
        self.requires("stb/cci.20210713")
        self.requires("nodesoup/cci.20200905")

    def imports(self):
        self.copy(src="./res/bindings", pattern="imgui_impl_glfw.*", dst="imgui_bindings", root_package='imgui')
        self.copy(src="./res/bindings", pattern="imgui_impl_opengl3*", dst="imgui_bindings", root_package='imgui')
        




    def configure(self):
        self.options['traact_core'].shared = self.options.shared
        self.options['traact_facade'].shared = self.options.shared
        self.options['traact_spatial'].shared = self.options.shared





