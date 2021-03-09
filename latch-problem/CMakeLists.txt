project("sentence_training")
CMAKE_MINIMUM_REQUIRED(VERSION 3.19)

set(CMAKE_CXX_FLAGS "-std=c++17")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

include_directories(./)
include_directories(./dynet)

configure_file(./dynet/libdynet.dylib libdynet.dylib COPYONLY)

add_executable(rnn rnn.cc)
target_link_libraries(rnn -L./ -ldynet)

