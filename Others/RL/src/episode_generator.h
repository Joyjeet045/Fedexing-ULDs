#pragma once
#include "mcts.h"
#include "container.h"

auto generate_episode(const Container&, int, float, int, int, size_t, std::vector<std::string>)
  -> std::vector<mcts::Evaluation<Container>>;

float calculate_baseline_reward(Container, std::vector<std::string>);