cmake_minimum_required(VERSION 3.26)
project(chat_with_audio LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(_dsp cpp/bindings.cpp)

install(TARGETS _dsp DESTINATION chat_with_audio)
