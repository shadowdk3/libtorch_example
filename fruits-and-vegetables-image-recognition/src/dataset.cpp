#include "dataset.h"
#include "log.h"

extern Logger logger;  // Declare the logger object defined in log.cpp

namespace {

    constexpr int kRows = 224;
    constexpr int kCols = 224;

    torch::Tensor CVtoTensor(cv::Mat img) {
        cv::resize(img, img, cv::Size{kRows, kCols}, 0, 0, cv::INTER_LINEAR);
        cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
        auto img_tensor = torch::from_blob(img.data, {kRows, kCols, 3}, torch::kByte);
        img_tensor = img_tensor.permute({2, 0, 1}).toType(torch::kFloat).div_(255);
        return img_tensor;
    }

    std::pair<torch::Tensor, torch::Tensor> read_data(const std::string& root,
                                                  bool train) {
        int i = 0;
        std::string ext(".jpg");
        const auto folder = train ? root + "/train" : root + "/test";
        std::vector<std::string> folders;

        std::string message = train ? "Loading train dataset" : "Loading test dataset";
        logger.writeLog(message);

        int num_samples = 0;
        for (const auto& entry : std::filesystem::directory_iterator(folder)) {
            if (std::filesystem::is_directory(entry.status())) {
                std::string subfolder = folder + '/' + entry.path().filename().string();
                for (const auto& e_ : std::filesystem::directory_iterator(subfolder)) {
                    if (std::filesystem::is_regular_file(e_)) {
                        ++num_samples;
                    }
                }
                folders.push_back(subfolder);
            }
        }

        auto targets = torch::empty(num_samples, torch::kInt64);
        auto images = torch::empty({num_samples, 3, kRows, kCols}, torch::kFloat);

        int64_t label = 0;

        for (auto& f : folders) {
            int count = 1;
            int total = 1;
            for (const auto& p : std::filesystem::directory_iterator(f)) {
                message = "Error: Unable to read image: " + p.path().string();

                if (p.path().extension() == ext) {
                    cv::Mat img = cv::imread(p.path());
                    if (img.empty()) {
                        logger.writeLog(message);

                        continue;  // Skip to the next iteration
                    }
                    auto img_tensor = CVtoTensor(img);
                    images[i] = img_tensor;
                    targets[i] = torch::tensor(label, torch::kInt64);

                    i++;
                    count++;
                } else {
                    logger.writeLog(message);
                }
                total ++;
            }
            
            std::ostringstream result_msg;
            result_msg << "Number of image in " << f << " load into dataset: " << count << " / " << total;
            logger.writeLog(result_msg.str());

            label++;
        }

        return {images, targets};
    }
}  // namespace

CustomDataset::CustomDataset(const std::string& root, Mode mode) : mode_(mode) {
  auto data = read_data(root, mode == Mode::kTrain);

  images_ = std::move(data.first);
  targets_ = std::move(data.second);
}

torch::data::Example<> CustomDataset::get(size_t index) {
  return {images_[index], targets_[index]};
}

torch::optional<size_t> CustomDataset::size() const { return images_.size(0); }

bool CustomDataset::is_train() const noexcept { return mode_ == Mode::kTrain; }

const torch::Tensor& CustomDataset::images() const { return images_; }

const torch::Tensor& CustomDataset::targets() const { return targets_; }

