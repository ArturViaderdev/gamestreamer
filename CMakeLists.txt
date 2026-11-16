cmake_minimum_required(VERSION 3.16)
project(gamestream CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
include(GNUInstallDirs)

# BUILD_DAEMON=OFF permite compilar sólo la capa (p. ej. la versión de 32 bits)
option(BUILD_DAEMON "Compilar gamestreamd" ON)
option(BUILD_LAYER  "Compilar la capa Vulkan" ON)
set(LAYER_SUFFIX "64" CACHE STRING "Sufijo de la capa: 64 o 32")

find_package(PkgConfig REQUIRED)

if(BUILD_LAYER)
  # Sólo hacen falta las cabeceras de Vulkan: la capa no enlaza con libvulkan.
  find_path(VULKAN_INCLUDE vulkan/vk_layer.h REQUIRED)
  add_library(VkLayer_gamestream SHARED src/layer/gamestream_layer.cpp)
  set_target_properties(VkLayer_gamestream PROPERTIES
    PREFIX ""
    OUTPUT_NAME "VkLayer_gamestream_${LAYER_SUFFIX}"
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN ON)
  target_include_directories(VkLayer_gamestream PRIVATE ${VULKAN_INCLUDE})
  # libstdc++ estática: así funciona dentro del contenedor de Steam (pressure-vessel)
  target_link_options(VkLayer_gamestream PRIVATE
    -static-libstdc++ -static-libgcc -Wl,--no-undefined -Wl,-Bsymbolic -Wl,-z,nodelete)
  target_link_libraries(VkLayer_gamestream PRIVATE pthread dl)

  set(LAYER_LIB_PATH "${CMAKE_INSTALL_FULL_LIBDIR}/VkLayer_gamestream_${LAYER_SUFFIX}.so")
  configure_file(src/layer/gamestream_layer.json.in
                 ${CMAKE_BINARY_DIR}/VkLayer_gamestream_${LAYER_SUFFIX}.json @ONLY)
  install(TARGETS VkLayer_gamestream LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
  install(FILES ${CMAKE_BINARY_DIR}/VkLayer_gamestream_${LAYER_SUFFIX}.json
          DESTINATION ${CMAKE_INSTALL_DATADIR}/vulkan/implicit_layer.d)
endif()

if(BUILD_DAEMON)
  pkg_check_modules(FF REQUIRED IMPORTED_TARGET
    libavcodec libavformat libavutil)
  add_executable(gamestreamd src/daemon/gamestreamd.cpp)
  # NVENC y CUDA se cargan con dlopen; sólo usamos las cabeceras incluidas en third_party
  target_include_directories(gamestreamd PRIVATE third_party/nvenc)
  target_link_libraries(gamestreamd PRIVATE PkgConfig::FF pthread dl)
  install(TARGETS gamestreamd RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
  install(PROGRAMS scripts/gamestream DESTINATION ${CMAKE_INSTALL_BINDIR})
  configure_file(scripts/gamestreamd.service.in ${CMAKE_BINARY_DIR}/gamestreamd.service @ONLY)
  install(FILES ${CMAKE_BINARY_DIR}/gamestreamd.service DESTINATION lib/systemd/user)
  install(FILES gamestream.conf.example DESTINATION ${CMAKE_INSTALL_DATADIR}/doc/gamestream)
endif()
