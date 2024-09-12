#ifndef LOGGER_H
#define LOGGER_H

#include <iostream>
#include <fstream>
#include <ctime>
#include <string>

class Logger {
    private:
        std::string logFileName;

    public:
        Logger();

        void createLogFile(const std::string& filename);
        void writeLog(const std::string& message);
        void createFolderIfNotExists(const std::string& folderPath);
        void loggerInit();

};

#endif