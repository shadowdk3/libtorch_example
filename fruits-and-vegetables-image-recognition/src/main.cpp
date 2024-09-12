#include <torch/torch.h>
#include <torch/script.h>
#include <iostream>
#include <ctime>
#include <chrono>
#include <iomanip>

#include "utils.h"
#include "dataset.h"
#include "log.h"
#include "main.h"

extern Logger logger;  // Declare the logger object defined in log.cpp

void printModuleSummary(torch::jit::script::Module module) {
    logger.writeLog("Model Summary:");
    logger.writeLog("----------------");
    std::ostringstream message;
    for (const auto& parameter : module.named_parameters()) {
        message << "Parameter: " << parameter.name;
        logger.writeLog(message.str());
        message.str("");
        message << "Size: " << parameter.value.sizes();
        logger.writeLog(message.str());
        message.str("");
        message << "Requires Gradient: " << parameter.value.requires_grad() ? "Yes" : "No";
        logger.writeLog(message.str());
    }
    logger.writeLog("----------------");
}

int main() {
    std::string root = "../data";

    torch::manual_seed(1);

    torch::Device device = getDevice();
    std::string message = "training device: " + device.str();
    logger.writeLog(message);

    torch::jit::script::Module model = torch::jit::load("artifacts/resnet50_torch.pth");
    logger.writeLog("load model");
    printModuleSummary(model);
    model.to(device);
    model.eval();

    auto train_dataset = CustomDataset(root)
                        .map(torch::data::transforms::Normalize<>(
                            {0.485, 0.456, 0.406}, {0.229, 0.224, 0.225}))
                        .map(torch::data::transforms::Stack<>());

    auto train_loader =
        torch::data::make_data_loader<torch::data::samplers::RandomSampler>(
            std::move(train_dataset), 64);

    auto test_dataset = CustomDataset(root, CustomDataset::Mode::kTest)
                            .map(torch::data::transforms::Normalize<>(
                                {0.485, 0.456, 0.406}, {0.229, 0.224, 0.225}))
                            .map(torch::data::transforms::Stack<>());

    auto test_loader =
        torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
            std::move(test_dataset), 4);

    // for (auto& batch : *train_loader) {
    //     auto img = batch.data;
    //     auto labels = batch.target;
    //     std::cout << "label: " << labels[0] << std::endl;
    //     // auto out = TensortoCv(img[0]);
    //     // cv::imshow("Display window", out);
    //     // int k = cv::waitKey(0);  // Wait for a keystroke in the window

    //     break;
    // }

    return 0;
}
