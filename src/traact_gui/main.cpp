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

    traact::util::initLogging(spdlog::level::level_enum::trace);

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
