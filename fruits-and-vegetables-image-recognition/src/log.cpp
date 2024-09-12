#include "log.h"
#include <cstring>
#include <filesystem>

Logger logger;  // Define the logger object

Logger::Logger() {
    logger.loggerInit();
}

void Logger::loggerInit() {
    auto now = std::chrono::system_clock::now();
    std::time_t now_c = std::chrono::system_clock::to_time_t(now);
    std::tm now_tm = *std::localtime(&now_c);
    std::ostringstream oss;
    oss << std::put_time(&now_tm, "%Y%m%d_%H_%M_%S");
    oss.str();

    std::string log_filename = "./log/" + oss.str() + ".txt";

    createLogFile(log_filename);
}

void Logger::createFolderIfNotExists(const std::string& folderPath) {
    if (!std::filesystem::exists(folderPath)) {
        std::filesystem::create_directory(folderPath);
        std::cout << "Folder created: " << folderPath << std::endl;
    } else {
        std::cout << "Folder already exists: " << folderPath << std::endl;
    }
}

void Logger::createLogFile(const std::string& log_filename) {
    createFolderIfNotExists("./log");
    logFileName = log_filename;
    
    std::ofstream logFile(logFileName, std::ios::app); // Open in append mode with the stored filename
    if (!logFile.is_open()) {
        std::cerr << "Error opening log file" << std::endl;
    }
    logFile.close();
}

void Logger::writeLog(const std::string& message) {
    std::ofstream logFile(logFileName, std::ios::app); // Open in append mode with the stored filename
    if (logFile.is_open()) {
        time_t now = time(0);
        char* dt = ctime(&now);
        dt[strlen(dt) - 1] = '\0'; // Remove the newline character at the end
        logFile << "[ " << dt << " ]" << " - " << message << std::endl;
        
        logFile.close();
    } else {
        std::cerr << "Error opening log file" << std::endl;
    }
}