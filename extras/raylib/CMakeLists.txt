cmake_minimum_required(VERSION 3.14)
project(raylib_clib_fetch NONE)

include(FetchContent)
set(FETCHCONTENT_QUIET OFF)

# A release tarball: faster than cloning raylib's history, and checked.
FetchContent_Declare(
  raylib
  URL https://github.com/raysan5/raylib/archive/refs/tags/6.0.tar.gz
  URL_HASH SHA256=2b3ee1e2120c7a0796b33062c7e9a694dd8a8caa56a96319ac8c8ecf54a90d0b
  DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)

set(RAYLIB_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(RAYLIB_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
# Both of GLFW's Linux backends: it uses Wayland when the program runs under it, else X11, and
# loads either's libraries itself when the program starts, so neither is linked.
set(GLFW_BUILD_WAYLAND ON CACHE BOOL "" FORCE)
set(GLFW_BUILD_X11 ON CACHE BOOL "" FORCE)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")

FetchContent_MakeAvailable(raylib)
