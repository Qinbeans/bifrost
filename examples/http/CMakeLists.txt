# Builds the C side of this project into build/lib, which config.yaml links:
#   libh2o-evloop.a   h2o (HTTP/1.1, HTTP/2, HTTP/3), fetched at a pinned commit
#   libbifrost_http.a c/http.c, the fasthttp-style API Bifrost calls
#
#   cmake -B build -G Ninja && cmake --build build --target bifrost_http
cmake_minimum_required(VERSION 3.14)
project(bifrost_http C)

include(FetchContent)
set(FETCHCONTENT_QUIET OFF)

FetchContent_Declare(
  h2o
  GIT_REPOSITORY https://github.com/h2o/h2o.git
  # h2o has no tagged release with HTTP/3; this is master of 2026-09-10.
  GIT_TAG cac7e6568ad98a848f099ecd0a18b881f632479a
)

set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(WITH_MRUBY OFF CACHE BOOL "" FORCE)
set(WITH_BROTLI OFF CACHE BOOL "" FORCE)
set(WITH_ZSTD OFF CACHE BOOL "" FORCE)
set(WITH_DTRACE OFF CACHE BOOL "" FORCE)
set(WITH_IO_URING OFF CACHE BOOL "" FORCE)
set(WITH_CCACHE OFF CACHE BOOL "" FORCE)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")

FetchContent_MakeAvailable(h2o)

add_library(bifrost_http STATIC c/http.c)
add_dependencies(bifrost_http libh2o-evloop)
target_compile_definitions(bifrost_http PRIVATE _GNU_SOURCE H2O_USE_LIBUV=0)
target_compile_options(bifrost_http PRIVATE -Wall -Wextra -Wshadow)
target_include_directories(bifrost_http PRIVATE
  ${h2o_SOURCE_DIR}/include
  ${h2o_SOURCE_DIR}/deps/picotls/include
  ${h2o_SOURCE_DIR}/deps/quicly/include
  ${h2o_SOURCE_DIR}/deps/klib
  ${h2o_SOURCE_DIR}/deps/picohttpparser
  ${h2o_SOURCE_DIR}/deps/cloexec
  ${h2o_SOURCE_DIR}/deps/golombset
  ${h2o_SOURCE_DIR}/deps/libgkc
  ${h2o_SOURCE_DIR}/deps/yoml
  ${h2o_BINARY_DIR}
)
