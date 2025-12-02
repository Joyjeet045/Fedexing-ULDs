#include "episode_generator.h"
#include <cpr/cpr.h>
#include <cstring>
#include <queue>
#include <mutex>
#include <future>

using mcts::InferenceResult_t;
template <typename State>
class InferenceQueue {
public:
  InferenceQueue(size_t batch_size, const std::vector<std::string>& _addresses)
   : m_batch_size { batch_size }, addresses { _addresses }, worker_use_count(addresses.size()) {}

  size_t size() noexcept { 
    std::lock_guard<std::mutex> lock_guard { inference_queue_mutex };
    return inference_queue.size();
   }

  size_t batch_size() const noexcept { return m_batch_size; }

  auto enqueue(const State& state) -> std::future<InferenceResult_t> {
    std::promise<InferenceResult_t> promise;
    auto future = promise.get_future();
    bool to_flush;
    {
      std::lock_guard<std::mutex> lock_guard { inference_queue_mutex };
      inference_queue.emplace(state, std::move(promise));
      to_flush = (inference_queue.size() >= m_batch_size);
    }

    if (to_flush) {
      flush();
    }

    return future;
  }

  void flush() {
    std::string data;
    std::vector<std::promise<InferenceResult_t>> promises;
    {
      std::lock_guard<std::mutex> lock_guard { inference_queue_mutex };
      if (inference_queue.empty()) return;
      for (size_t i = 0; i < m_batch_size && !inference_queue.empty(); ++i) {
        auto [state, promise] = std::move(inference_queue.front());
        inference_queue.pop();

        data += state.serialize();
        promises.emplace_back(std::move(promise));
      }
    }

    cpr::Url url { "http://" + get_address() + "/policy_value_inference" };
    cpr::Header header {
      { "batch-size", std::to_string(promises.size()) },
      { "Content-Type", "application/octet-stream" }
    };
    cpr::Body body { &data[0], data.size() };
    auto response = cpr::Post(url, header, body);
    auto text = response.text;
    const char* ptr = &text[0];
    for (auto& promise : promises) {
      std::vector<float> priors(State::action_count);
      float value;

      std::memcpy(&priors[0], ptr, sizeof(float) * State::action_count);
      ptr += sizeof(float) * State::action_count;
      std::memcpy(&value, ptr, sizeof(float));
      ptr += sizeof(float);

      promise.set_value(std::make_pair(priors, value));
    }
  }

private:
  std::string get_address() noexcept {
    std::lock_guard<std::mutex> lock_guard { worker_use_count_mutex };

    size_t min_idx = 0;
    for (size_t i = 0; i < addresses.size(); ++i) {
      if (worker_use_count[i] < worker_use_count[min_idx]) {
        min_idx = i;
      }
    }

    ++worker_use_count[min_idx];
    return addresses[min_idx];
  }

  std::mutex inference_queue_mutex, worker_use_count_mutex;
  size_t m_batch_size;
  std::queue<std::pair<State, std::promise<InferenceResult_t>>> inference_queue;
  std::vector<std::string> addresses;
  std::vector<size_t> worker_use_count;
};

auto generate_episode(
    const Container& container, int simulations_per_move, 
    float c_puct, int virtual_loss, int thread_count, size_t batch_size, std::vector<std::string> addresses)
      -> std::vector<mcts::Evaluation<Container>> {

  InferenceQueue<Container> inference_queue { batch_size, addresses };
  auto episodes = mcts::generate_episode(container, simulations_per_move, c_puct, virtual_loss, thread_count, inference_queue);
  return episodes;
}

float calculate_baseline_reward(Container container, std::vector<std::string> addresses) {
  InferenceQueue<Container> inference_queue { 1, addresses };
  while (true) {
    auto result = inference_queue.enqueue(container);
    auto possible_actions = container.possible_actions();

    inference_queue.flush();
    auto priors = result.get().first;

    if (possible_actions.empty()) break;
    int max_prior_action = possible_actions[0];
    for (auto action_idx : possible_actions) {
      if (priors[action_idx] > priors[max_prior_action]) {
        max_prior_action = action_idx;
      }
    }
    container.transition(max_prior_action);
  }
  return container.reward();
}