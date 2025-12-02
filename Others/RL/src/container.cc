#include "container.h"
#include <map>
#include <cassert>
#include <cstring>
#include <algorithm>

glm::ivec3 get_shape_along_axes(glm::ivec3 shape, int orientation) {
  static constexpr int ORIENTATIONS[6][3] = {
    { 0, 1, 2 },
    { 0, 2, 1 },
    { 1, 0, 2 },
    { 1, 2, 0 },
    { 2, 0, 1 },
    { 2, 1, 0 },
  };
  
  glm::ivec3 shape_along_axes;
  for (int i = 0; i < 3; ++i) {
    shape_along_axes[i] = shape[ORIENTATIONS[orientation][i]];
  }
  return shape_along_axes;
}

template <typename T>
auto get_max_freq_in_window(const Array2D<T>& arr, glm::ivec3 shape) -> Array2D<std::pair<T, int>> {
  std::size_t n_arr = arr.rows(), m_arr = arr.cols();
  std::size_t n_shape = shape.x, m_shape = shape.y;

  Array2D<std::pair<T, int>> res { n_arr, m_arr };
  for (std::size_t x = 0; x < n_arr; ++x) {
    std::map<int, int, std::greater<>> freq;
    for (std::size_t y = 0; y < m_shape; ++y) {
      ++freq[arr(x, y)];
    }

    for (std::size_t y = 0; y < m_arr - m_shape; ++y) {
      res(x, y) = *freq.begin();

      int val = arr(x, y);
      if (--freq[val] == 0) {
        freq.erase(val);
      }

      ++freq[arr(x, y + m_shape)];
    }
    res(x, m_arr - m_shape) = *freq.begin();
  }

  for (std::size_t y = 0; y <= m_arr - m_shape; ++y) {
    std::map<int, int, std::greater<>> freq;
    for (std::size_t x = 0; x < n_shape; ++x) {
      freq[res(x, y).first] += res(x, y).second;
    }

    for (std::size_t x = 0; x < n_arr - n_shape; ++x) {
      auto [val, cnt] = res(x, y);
      res(x, y) = *freq.begin();

      freq[val] -= cnt;
      if (freq[val] == 0) {
        freq.erase(val);
      }

      freq[res(x + n_shape, y).first] += res(x + n_shape, y).second;
    }
    res(n_arr - n_shape, y) = *freq.begin();
  }

  return res;
}

Container::Container(int height, const std::vector<Package>& packages)
  : m_height { height }, m_packages { packages },
    m_height_map { Container::length, Container::width } {}

Container::Container(int height, std::vector<Package>&& packages, Array2D<int>&& height_map)
  : m_height { height }, m_packages { std::move(packages) }, m_height_map { std::move(height_map) } {}

auto Container::height() const noexcept -> int {
  return m_height;
}

auto Container::packages() const noexcept -> const std::vector<Package>& {
  return m_packages;
}

auto Container::height_map() const noexcept -> const Array2D<int>& {
  return m_height_map;
}

auto Container::possible_actions() const -> std::vector<int> {
  std::vector<int> actions;
  auto mask = get_valid_state_mask(m_packages.front());
  for (int x = 0; x < mask.rows(); ++x) {
    for (int y = 0; y < mask.cols(); ++y) {
      if (mask(x, y) >= 0) {
        actions.push_back(x * mask.cols() + y);
      }
    }
  }

  return actions;
}

void Container::transition(int action_idx) {
  assert(action_idx >= 0 && action_idx < Container::action_count);
  auto shape = m_packages.front().shape;
  m_packages.front().is_placed = true;
  std::rotate(m_packages.begin(), m_packages.begin() + 1, m_packages.end());
  int x0 = action_idx / Container::width, y0 = action_idx % Container::width;
  for (int x = x0; x < x0 + shape.x; ++x) {
    for (int y = y0; y < y0 + shape.y; ++y) {
      m_height_map(x, y) += shape.z;
      assert(m_height_map(x, y) <= m_height);
    }
  }
}

float Container::reward() const noexcept {
  float total_volume = 0.0f;
  for (auto pkg : m_packages) {
    if (pkg.is_placed) total_volume += pkg.shape.x * pkg.shape.y * pkg.shape.z;
  }
  float packing_efficiency = total_volume / (m_height * Container::length * Container::width);
  return packing_efficiency;
}

auto Container::serialize() const noexcept -> std::string {
  std::pair<const void*, size_t> infos[] = {
    { &m_height, sizeof(int) },
    { &m_packages[0], sizeof(Package) * package_count },
    { &m_height_map(0, 0), sizeof(int) * length * width }
  };

  size_t total_size = 0;
  for (auto [ptr, size] : infos) {
    total_size += size;
  }

  std::string bytes(total_size, ' ');
  char* dest = &bytes[0];
  for (auto [src, size] : infos) {
    std::memcpy(dest, src, size);
    dest += size;
  }
  return bytes;
}

Container Container::unserialize(const std::string& bytes) {
  int height;
  std::vector<Package> packages(package_count);
  Array2D<int> height_map(length, width);

  std::pair<void*, size_t> infos[] = {
    { &height, sizeof(int) },
    { &packages[0], sizeof(Package) * package_count },
    { &height_map(0, 0), sizeof(int) * length * width }
  };

  const char* src = &bytes[0];
  for (auto [dest, size] : infos) {
    std::memcpy(dest, src, size);
    src += size;
  }

  return Container(height, std::move(packages), std::move(height_map));
}

auto Container::get_valid_state_mask(const Package& pkg) const noexcept -> Array2D<int> {
  Array2D<int> mask { Container::length, Container::width, -1 };
  auto max_height_freq = get_max_freq_in_window(m_height_map, pkg.shape);
  for (std::size_t x = 0; x <= mask.rows() - pkg.shape.x; ++x) {
    for (std::size_t y = 0; y <= mask.cols() - pkg.shape.y; ++y) {
      auto [max_height, freq] = max_height_freq(x, y);
      if (max_height + pkg.shape.z > m_height) continue;

      float min_base_contact_ratio = 0.50f;
      float base_contact_ratio = static_cast<float>(freq) / (pkg.shape.x * pkg.shape.y);
      if (base_contact_ratio >= min_base_contact_ratio) {
        mask(x, y) = max_height;
      }
    }
  }
  return mask;
}