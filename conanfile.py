# /usr/bin/python3
import os
from conan import ConanFile
from conan.tools.build import can_run
from conan.tools.files import copy

class TraactPackage(ConanFile):
    python_requires = "traact_base/0.0.0@traact/latest"
    python_requires_extend = "traact_base.TraactPackageCmake"

    name = "traact_gui"
    version = "0.0.0"    
    description = "GUI Editor for Traact"
    url = "https://github.com/traact/traact.git"
    license = "MIT"
    author = "Frieder Pankratz"    

    settings = "os", "compiler", "build_type", "arch"
    compiler = "cppstd"

    exports_sources = "CMakeLists.txt", "src/*", "external/*",

    keep_imports = True
    options = {
        "shared": [True, False],
    }

    default_options = {
        "shared": True,
        "magnum/*:with_anyimageimporter": True,
        "magnum/*:with_tgaimporter": True,
        "magnum/*:with_anysceneimporter": True,
        "magnum/*:with_gl_info": True,
        "magnum/*:with_objimporter": True,
        "magnum/*:with_tgaimageconverter": True,
        "magnum/*:with_imageconverter": True,
        "magnum/*:with_anyimageconverter": True,
        "magnum/*:with_sdl2application": True,
        "magnum/*:with_eglcontext": False,
        "magnum/*:with_windowlesseglapplication": False,
        "magnum/*:target_gles": False,
        "magnum/*:with_opengltester": True,
        "magnum-integration/*:with_bullet": False,  # does not build on windows debug for the moment ...
        "magnum-integration/*:with_imgui": True,
        "magnum-integration/*:with_eigen": True,
        "magnum-plugins/*:with_stbimageimporter": True,
        "magnum-plugins/*:with_stbimageconverter": True,
    }    

    def requirements(self):
        self.requires("traact_base/0.0.0@traact/latest")
        self.requires("traact_core/0.0.0@traact/latest")
        self.requires("traact_spatial/0.0.0@traact/latest")
        self.requires("traact_vision/0.0.0@traact/latest")
        self.requires("cuda_dev_config/2.2@camposs/stable", override=True)
        self.requires("opencv/4.8.0@camposs/stable")
        self.requires("glfw/3.3.8")
        self.requires("glew/2.2.0")
        self.requires("imgui/cci.20230105+1.89.2.docking", override=True)
        self.requires("ghc-filesystem/1.5.8")
        self.requires("imguizmo/1.83")
        self.requires("implot/0.16")
        self.requires("stb/cci.20220909")
        self.requires("nodesoup/cci.20200905")
        self.requires("open3d/0.17.0@camposs/stable")
        self.requires("glm/0.9.9.8")
        self.requires("pcpd_shm_client/0.4.0@artekmed/stable", run=True)
        self.requires("openssl/1.1.1t", force=True)
        #self.requires("magnum/2020.06@camposs/stable")
        #self.requires("corrade/2020.06@camposs/stable")
        #self.requires("magnum-integration/2020.06@camposs/stable")
        #self.requires("magnum-plugins/2020.06@camposs/stable")

        self.requires("re2/20230801")

        self.requires("traact_pointcloud/0.0.0@traact/latest")
        
        #self.requires("traact_component_basic/1.0.0@traact/latest")
        #self.requires("traact_component_kinect_azure/1.0.0@traact/latest")
        #self.requires("traact_component_cereal/1.0.0@traact/latest")
        #self.requires("traact_component_aruco/1.0.0@traact/latest")
        #self.requires("traact_component_pcpd_shm/1.0.0@traact/latest")
        
        #self.requires("traact_component_http/1.0.0@traact/latest")

        
        


    def _extend_generate(self):            
        for dep in self.dependencies.values():            
            if dep.ref.name == "imgui":
                copy(self, "imgui_impl_opengl3*", os.path.join(dep.package_folder, "res/bindings"), os.path.join(self.build_folder, "imgui_bindings"))
                copy(self, "imgui_impl_glfw*", os.path.join(dep.package_folder, "res/bindings"), os.path.join(self.build_folder, "imgui_bindings"))                
        

    def configure(self):
        self.options['traact_core'].shared = self.options.shared        
        self.options['traact_spatial'].shared = self.options.shared
        self.options['traact_vision'].shared = self.options.shared
        self.options['glfw'].shared = self.options.shared
        self.options['glew'].shared = self.options.shared
        self.options['pcpd_shm_client'].with_python = False
        self.options['pcpd_shm_client'].with_visualization = False
        self.options['pcpd_shm_client'].with_apps = False
        self.options['pcpd_shm_client'].shared = False
        #self.options['tcn_schema'].with_dds = True
        #self.options['open3d'].with_visualization = True






