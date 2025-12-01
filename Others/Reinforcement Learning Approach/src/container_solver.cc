#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "episode_generator.h"
namespace py = pybind11;

template <typename T>
py::buffer_info array_2d_buffer_info(Array2D<T>& arr) {
  py::buffer_info info{};
  info.ptr = static_cast<void*>(arr.data());
  info.itemsize = sizeof(T);
  info.format = py::format_descriptor<T>::format();
  info.ndim = 2;
  info.shape = { static_cast<long>(arr.rows()), static_cast<long>(arr.cols()) };
  info.strides = { static_cast<long>(sizeof(T) * arr.cols()), sizeof(T) };
  return info;
}

PYBIND11_MAKE_OPAQUE(Container)

PYBIND11_MODULE(container_solver, m) {
  // glm::vec3
  py::class_<glm::ivec3>(m, "Vec3i")
    .def(py::init<int, int, int>())
    .def_readwrite("x", &glm::ivec3::x)
    .def_readwrite("y", &glm::ivec3::y)
    .def_readwrite("z", &glm::ivec3::z);

  // Package
  py::class_<Package>(m, "Package")
    .def(py::init<>())
    .def_readwrite("shape", &Package::shape)
    .def_readwrite("weight", &Package::weight)
    .def_readwrite("is_priority", &Package::is_priority)
    .def_readwrite("cost", &Package::cost)
    .def_readwrite("is_placed", &Package::is_placed)
    .def_readwrite("pos", &Package::pos);

  // Array2D<int>
  py::class_<Array2D<int>>(m, "Array2Di", py::buffer_protocol())
    .def_buffer(array_2d_buffer_info<int>);
  
  // Container
  py::class_<Container>(m, "Container")
    .def(py::init<int, std::vector<Package>>(), py::arg("height"), py::arg("packages"))

    .def_property_readonly("height", &Container::height)
    .def_property_readonly("packages", &Container::packages)
    .def_property_readonly("height_map", &Container::height_map)

    .def_property_readonly("possible_actions", &Container::possible_actions)
    .def("transition", &Container::transition, py::arg("action_idx"))
    .def_property_readonly("reward", &Container::reward)

    .def("serialize", &Container::serialize)
    .def("unserialize", &Container::unserialize)

    .def_readonly_static("action_count", &Container::action_count)
    .def_readonly_static("package_count", &Container::package_count)
    .def_readonly_static("length", &Container::length)
    .def_readonly_static("width", &Container::width)

    .def(py::pickle(
      [] (const Container& container) { return container.serialize(); }, 
      &Container::unserialize
    ));

  // Evaluation
  using Evaluation = mcts::Evaluation<Container>;
  py::class_<Evaluation>(m, "Evaluation")
    .def_readwrite("container", &Evaluation::state)
    .def_readwrite("action_idx", &Evaluation::action_idx)
    .def_readwrite("priors", &Evaluation::priors)
    .def_readwrite("reward", &Evaluation::reward);

  // generate episode
  using ContainerEvaluatorFn = const std::function<std::pair<std::vector<float>, float>(const Container&)>&;
  m.def(
    "generate_episode",
    &generate_episode,
    py::arg("container"),
    py::arg("simulations_per_move"),
    py::arg("c_puct"),
    py::arg("virtual_loss"),
    py::arg("thread_count"),
    py::arg("batch_size"),
    py::arg("addresses")
  );

  m.def(
    "calculate_baseline_reward",
    &calculate_baseline_reward,
    py::arg("container"),
    py::arg("addresses")
  );
}