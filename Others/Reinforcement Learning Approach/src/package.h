#pragma once
#include <glm/vec3.hpp>

struct Package {
  glm::ivec3 shape;
  int weight;
  bool is_priority;
  int cost;

  bool is_placed;
  glm::ivec3 pos;
};