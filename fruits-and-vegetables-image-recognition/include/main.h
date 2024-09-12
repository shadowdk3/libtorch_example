#ifndef MAIN_H
#define MAIN_H

#include "log.h"

void printModuleSummary(Logger& logger, torch::jit::script::Module module);

#endif