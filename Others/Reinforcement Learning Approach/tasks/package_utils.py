from random import random, randint
from container_solver import Container, Package, Vec3i
import numpy as np

def random_package():
  dims_range = (4, 8)
  pkg = Package()
  shape = (randint(*dims_range) for _ in range(3))
  pkg.shape = Vec3i(*shape)
  return pkg

def normalize_packages(container: Container):
  packages = container.packages
  data = np.zeros(4 * len(packages), dtype=np.float32)
  for i, pkg in enumerate(packages):
    if pkg.is_placed:
      continue

    pkg_x = pkg.shape.x / Container.length
    pkg_y = pkg.shape.y / Container.width
    pkg_z = pkg.shape.z / container.height
    data[4*i : 4*(i+1)] = [
      pkg_x,
      pkg_y,
      pkg_z,
      pkg_x * pkg_y * pkg_z
    ]
  
  return data
