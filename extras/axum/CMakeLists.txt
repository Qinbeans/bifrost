# Builds the C side of axum into build/lib, which config.yaml links:
#   libh2o-evloop.a          h2o (HTTP/1.1, HTTP/2, HTTP/3), fetched at a pinned commit
#   libbifrost_axum.a        c/axum.c, c/stream.c and c/defaults.c: routers, middleware, requests, responses, the server
#   libaxum_ws.a, libaxum_multipart.a   c/ws.c and c/multipart.c, for the ws and multipart features
#   axum_feature_<name>.o    one per feature the C side reads (http1, http2, tracing, matched_path,
#                            original_uri): linked only when that feature is on (see config.yaml)
#
#   cmake -B build -G Ninja && cmake --build build
cmake_minimum_required(VERSION 3.28)
project(bifrost_axum C)

include(FetchContent)
set(FETCHCONTENT_QUIET OFF)

FetchContent_Declare(
  h2o
  # h2o has no tagged release with HTTP/3; this is master of 2026-09-10, as a tarball.
  URL https://github.com/h2o/h2o/archive/cac7e6568ad98a848f099ecd0a18b881f632479a.tar.gz
  URL_HASH SHA256=be4a7792211ab0d70513e5a2eea4e360b223dd1f8e43af4eacac214d1fba011b
  DOWNLOAD_EXTRACT_TIMESTAMP TRUE
  # Only what axum needs (libh2o-evloop) builds: not h2o's server, tools and tests.
  EXCLUDE_FROM_ALL
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

set(H2O_INCLUDES
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

add_library(bifrost_axum STATIC c/axum.c c/stream.c c/defaults.c)
add_dependencies(bifrost_axum libh2o-evloop)
target_compile_definitions(bifrost_axum PRIVATE _GNU_SOURCE H2O_USE_LIBUV=0)
target_compile_options(bifrost_axum PRIVATE -Wall -Wextra -Wshadow)
target_include_directories(bifrost_axum PRIVATE ${H2O_INCLUDES})

# The features the C side reads: an object each, defining its flag (linked as an object, so always taken).
set(FEATURE_OBJECTS "")
foreach(feature http1 http2 tracing matched_path original_uri)
  set(object "${CMAKE_BINARY_DIR}/lib/axum_feature_${feature}.o")
  add_custom_command(
    OUTPUT ${object}
    COMMAND ${CMAKE_C_COMPILER} -c -fPIC -o ${object} ${CMAKE_SOURCE_DIR}/c/features/${feature}.c
    DEPENDS ${CMAKE_SOURCE_DIR}/c/features/${feature}.c
  )
  list(APPEND FEATURE_OBJECTS ${object})
endforeach()
add_custom_target(axum_features ALL DEPENDS ${FEATURE_OBJECTS})
add_dependencies(bifrost_axum axum_features)

# Features with C of their own: a library each, linked only when the feature is on.
foreach(feature ws multipart)
  add_library(axum_${feature} STATIC c/${feature}.c)
  add_dependencies(axum_${feature} libh2o-evloop)
  target_compile_definitions(axum_${feature} PRIVATE _GNU_SOURCE H2O_USE_LIBUV=0)
  target_compile_options(axum_${feature} PRIVATE -Wall -Wextra -Wshadow)
  target_include_directories(axum_${feature} PRIVATE ${H2O_INCLUDES})
endforeach()
