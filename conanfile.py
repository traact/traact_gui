# /usr/bin/python3
import os
from conans import ConanFile, CMake, tools


class Traact(ConanFile):
    name = "traact_gui"
    version = "0.0.1"
    
    description = "GUI Editor for Traact"
    url = ""
    license = "BSD 3-Clause"
    author = "Frieder Pankratz"

    short_paths = True

    generators = "cmake", "traact_virtualrunenv_generator"
    settings = "os", "compiler", "build_type", "arch"
    compiler = "cppstd"
    keep_imports = True
    options = {
        "shared": [True, False],
    }

    default_options = {
        "shared": True,        
    }

    exports_sources = "CMakeLists.txt", "main.cpp"

    def requirements(self):        
        self.requires("traact_run_env/%s@camposs/stable" % self.version)
        self.requires("traact_core/%s@camposs/stable" % self.version)
        self.requires("traact_spatial/%s@camposs/stable" % self.version)
        self.requires("traact_vision/%s@camposs/stable" % self.version)
        self.requires("traact_serialization/%s@camposs/stable" % self.version)
        self.requires("nlohmann_json/3.7.3")
        self.requires("glfw/3.3.4")
        self.requires("glew/2.2.0")
        self.requires("imgui/1.83")
        self.requires("ghc-filesystem/1.5.8")
        #self.requires("imguizmo/cci.20210720")
        self.requires("implot/0.11")
        self.requires("stb/cci.20210713")
        self.requires("nodesoup/cci.20200905")

    def imports(self):
        self.copy(src="./res/bindings", pattern="imgui_impl_glfw.*", dst="imgui_bindings", root_package='imgui')
        self.copy(src="./res/bindings", pattern="imgui_impl_opengl3.*", dst="imgui_bindings", root_package='imgui')
        




    def configure(self):
        self.options['traact_core'].shared = self.options.shared
        self.options['traact_facade'].shared = self.options.shared
        self.options['traact_spatial'].shared = self.options.shared
        #self.options['traact_vision'].shared = self.options.shared
        #self.options['traact_kinect_azure'].with_bodytracking = True





