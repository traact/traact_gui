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
        




    def configure(self):
        self.options['traact_core'].shared = self.options.shared
        self.options['traact_facade'].shared = self.options.shared
        self.options['traact_spatial'].shared = self.options.shared
        #self.options['traact_vision'].shared = self.options.shared
        #self.options['traact_kinect_azure'].with_bodytracking = True





