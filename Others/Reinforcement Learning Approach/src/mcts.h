#pragma once
#include <cmath>
#include <atomic>
#include <thread>
#include <vector>
#include <future>
#include <ranges>
#include <utility>
#include <semaphore>

namespace mcts {

template <typename State>
concept StateConcept = requires(State& state, int action_idx) {
  { State::action_count } -> std::convertible_to<size_t>;
  { std::as_const(state).possible_actions() } -> std::same_as<std::vector<int>>;
  { state.transition(action_idx) } -> std::same_as<void>;
  { std::as_const(state).reward() } -> std::same_as<float>;
};

using InferenceResult_t = std::pair<std::vector<float>, float>;
template <typename InferenceQueue, typename State>
concept InferenceQueueConcept = requires(const State& state, InferenceQueue& inference_queue) {
  { std::as_const(inference_queue).batch_size() } -> std::same_as<size_t>;
  { inference_queue.enqueue(state) } -> std::same_as<std::future<InferenceResult_t>>;
  { inference_queue.flush() } -> std::same_as<void>;
  { inference_queue.size() } -> std::same_as<size_t>;
  { std::as_const(inference_queue).batch_size() } -> std::same_as<size_t>;
};

template <StateConcept State>
struct Node {
  Node() = default;

  std::unique_ptr<State> state = nullptr;
  std::weak_ptr<Node> prev_node;
  
  std::atomic<bool> visited = false, evaluated = false;
  std::atomic<int> unvisited_leaf_count = 1;
  std::atomic<int> action_idx = -1;
  std::atomic<int> visit_count = 0;
  std::atomic<float> total_action_value = 0.0f;
  std::atomic<float> prior = 0.0f;

  using Ptr = std::shared_ptr<Node<State>>;
  std::vector<Ptr> children;
};

template <typename State>
using NodePtr = Node<State>::Ptr;

template <StateConcept State, InferenceQueueConcept<State> InferenceQueue>
bool run_mcts_simulation(
  NodePtr<State> node, float c_puct, int virtual_loss, 
  InferenceQueue& inference_queue, 
  std::atomic<int>& simulation_count, int max_simulations) {
  // Selection
  std::vector<NodePtr<State>> search_path { node };
  while (node->visited) {
    if (!node->evaluated) return false;
    if (node->children.empty()) return true;

    int total_visit_count = node->visit_count - 1;
    auto calculate_score = [c_puct, total_visit_count] (NodePtr<State> node) {
      float mean_action_value = (node->visit_count > 0 ? node->total_action_value / node->visit_count : 0);
      float puct_score = c_puct * node->prior * sqrt(total_visit_count) / (1 + node->visit_count);
      float score = mean_action_value + puct_score;
      return score;
    };

    float max_score = -std::numeric_limits<float>::infinity();
    NodePtr<State> next_node;
    for (auto child : node->children) {
      float score = calculate_score(child);
      if (score > max_score) {
        max_score = score;
        next_node = child;
      }
    }

    node = next_node;
    search_path.push_back(node);
  }

  auto old_visited_value = node->visited.exchange(true);
  if (old_visited_value) return false;
  
  // Apply virtual loss
  for (auto node : search_path) {
    node->visit_count += virtual_loss;
    node->total_action_value -= virtual_loss;
  }

  // Lazy State Update
  if (node->state == nullptr) {
    auto prev_node = node->prev_node.lock();
    node->state = std::make_unique<State>(*prev_node->state);
    node->state->transition(node->action_idx);
  }

  // Enqueue for async evaluation
  auto evaluator_result = inference_queue.enqueue(*node->state);
  if (search_path[0]->unvisited_leaf_count <= inference_queue.batch_size()) {
    inference_queue.flush();
  }

  // Expansion
  auto actions = node->state->possible_actions();
  for (auto action_idx : actions) {
    auto child = std::make_shared<Node<State>>();
    child->prev_node = node;
    child->action_idx = action_idx;
    node->children.push_back(child);
  }

  // Priors update
  auto [priors, value] = evaluator_result.get();
  float total_valid_prior = 0.0f;
  for (auto action_idx : actions) {
    total_valid_prior += priors[action_idx];
  }

  for (auto child : node->children) {
    child->prior = priors[child->action_idx] / total_valid_prior;
  }

  // Backpropagation
  int action_count = actions.size();
  for (auto node : std::ranges::views::reverse(search_path)) {
    node->visit_count += 1 - virtual_loss;
    node->total_action_value += value + virtual_loss;
    node->unvisited_leaf_count += action_count - 1;
    node->evaluated = true;
  }

  return true;
}

template <typename State>
struct Evaluation {
  Evaluation (const State& _state, int _action_idx = -1, const std::vector<float>& _priors = {}, float _reward = {})
    : state { _state }, action_idx { _action_idx }, priors { _priors }, reward { _reward } {}

  State state;
  int action_idx;
  std::vector<float> priors;
  float reward;
};

template <StateConcept State, InferenceQueueConcept<State> InferenceQueue>
auto generate_episode(
  State state, int simulations_per_move, 
  float c_puct, int virtual_loss, int thread_count, 
  InferenceQueue& inference_queue)
    -> std::vector<Evaluation<State>> {

  std::atomic<bool> to_end = false;
  std::jthread inference_queue_daemon {[&] {
    while (!to_end) {
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
      if (inference_queue.size() < inference_queue.batch_size()) {
        inference_queue.flush();
      }
    }
  }};

  std::vector<Evaluation<State>> state_evaluations;
  auto node = std::make_shared<Node<State>>();
  node->state = std::make_unique<State>(state);
  while (true) {
    std::vector<std::thread> threads;
    std::counting_semaphore counting_semaphore { thread_count };
    std::atomic<int> simulation_count;
    while (simulation_count < simulations_per_move) {
      counting_semaphore.acquire();

      std::binary_semaphore launched_semaphore { 0 };
      auto thread = std::thread([&] {
        launched_semaphore.release();
        bool success = run_mcts_simulation<State>(
          node, c_puct, virtual_loss, 
          inference_queue, simulation_count, simulations_per_move
        );
        if (success) ++simulation_count;
        counting_semaphore.release();
      });

      launched_semaphore.acquire();
      threads.emplace_back(std::move(thread));
    }

    for (auto& thread : threads) {
      thread.join();
    }

    if (node->children.empty()) break;

    std::vector<float> priors(State::action_count);
    int max_visit_count = -1;
    int action_idx;
    NodePtr<State> next_node;
    for (auto child : node->children) {
      if (child->visit_count > max_visit_count) {
        max_visit_count = child->visit_count;
        action_idx = child->action_idx;
        next_node = child;
      }

      priors[child->action_idx] = static_cast<float>(child->visit_count) / (node->visit_count - 1);
    }

    state_evaluations.emplace_back(*node->state, action_idx, priors);
    next_node->state = std::make_unique<State>(*node->state);
    next_node->state->transition(action_idx);
    node = next_node;
  }

  auto reward = node->state->reward();
  for (auto& evaluation : state_evaluations) {
    evaluation.reward = reward;
  }

  to_end = true;
  return state_evaluations;
}

}