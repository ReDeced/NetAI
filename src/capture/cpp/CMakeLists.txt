cmake_minimum_required(VERSION 3.16)
project(netai_sniffer CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_compile_options(-O2 -Wall -Wextra)

find_library(PCAP_LIB pcap REQUIRED)

add_executable(sniffer sniffer.cpp)
target_link_libraries(sniffer ${PCAP_LIB})
