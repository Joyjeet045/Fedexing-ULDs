if (WIN32)
  set(BUILD_SHARED_LIBS OFF)
endif()

include(FetchContent)
FetchContent_Declare(
  glm
  GIT_REPOSITORY https://github.com/g-truc/glm.git
  GIT_TAG 0af55ccecd98d4e5a8d1fad7de25ba429d60e863
)
FetchContent_MakeAvailable(glm)

FetchContent_Declare(
  cpr
  GIT_REPOSITORY https://github.com/libcpr/cpr.git
  GIT_TAG dec9422db3af470641f8b0d90e4b451c4daebf64
)
FetchContent_MakeAvailable(cpr)

FetchContent_Declare(
  pybind11
  GIT_REPOSITORY https://github.com/pybind/pybind11.git
  GIT_TAG a2e59f0e7065404b44dfe92a28aca47ba1378dc4
)
FetchContent_MakeAvailable(pybind11)

pybind11_add_module(
  container_solver
  mcts.h
  array2d.h
  package.h
  container.h
  container.cc
  episode_generator.h
  episode_generator.cc
  container_solver.cc
)

target_compile_features(container_solver PUBLIC cxx_std_20)
target_link_libraries(container_solver PUBLIC glm::glm-header-only cpr::cpr)

set(MODULE_OUTPUT_DIR ${CMAKE_SOURCE_DIR}/src/container-solver)
set_target_properties(
  container_solver
  PROPERTIES LIBRARY_OUTPUT_DIRECTORY
  ${MODULE_OUTPUT_DIR}
)

set(STUBGEN_EXECUTABLE "${CMAKE_SOURCE_DIR}/venv/bin/stubgen")
if(WIN32)
  set(STUBGEN_EXECUTABLE "${CMAKE_SOURCE_DIR}/venv/Scripts/stubgen.exe")
endif()

if (EXISTS ${STUBGEN_EXECUTABLE})
  add_custom_command(TARGET container_solver POST_BUILD
    COMMAND ${STUBGEN_EXECUTABLE} -m container_solver -o .
    WORKING_DIRECTORY ${MODULE_OUTPUT_DIR})
endif()