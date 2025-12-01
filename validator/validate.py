#!/usr/bin/env python
import argparse
from dataclasses import dataclass

@dataclass
class ULD:
  length: int
  width: int
  height: int
  weight_limit: int

@dataclass
class Package:
  length: int
  width: int
  height: int
  weight: int
  priority: str
  cost: int

@dataclass
class PkgPosition:
  uld_id: int
  x_min: int
  y_min: int
  z_min: int
  x_max: int
  y_max: int
  z_max: int

class ConstraintException(Exception):
  pass

def check_intersection(pkg1: PkgPosition, pkg2: PkgPosition):
  x_intersect = pkg1.x_max > pkg2.x_min and pkg2.x_max > pkg1.x_min
  y_intersect = pkg1.y_max > pkg2.y_min and pkg2.y_max > pkg1.y_min
  z_intersect = pkg1.z_max > pkg2.z_min and pkg2.z_max > pkg1.z_min
  return x_intersect and y_intersect and z_intersect

def get_xy_overlap_area(pkg1: PkgPosition, pkg2: PkgPosition):
  """Calculates the area of overlap in the XY plane (top-down view)."""
  x_overlap = max(0, min(pkg1.x_max, pkg2.x_max) - max(pkg1.x_min, pkg2.x_min))
  y_overlap = max(0, min(pkg1.y_max, pkg2.y_max) - max(pkg1.y_min, pkg2.y_min))
  return x_overlap * y_overlap

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--input")
  parser.add_argument("--output")
  args = parser.parse_args()

  n = m = k = None
  ulds = None
  packages = None
  with open(args.input) as f:
    n, m, k = (int(x) for x in f.readline().split(','))
    ulds = [None for i in range(n + 1)]
    packages = [None for i in range(m + 1)]

    for i in range(1, n + 1):
      length, width, height, weight_limit = (int(x) for x in f.readline().split(',')[1:])
      ulds[i] = ULD(length, width, height, weight_limit)

    for i in range(1, m + 1):
      vals = f.readline().split(',')
      length, width, height, weight = (int(x) for x in vals[1:5])
      priority = vals[5]
      cost = int(vals[6]) if vals[6] != '-\n' else 0
      packages[i] = Package(length, width, height, weight, priority, cost)

  pkg_positions = [None for i in range(m + 1)]
  cost = used_pkg = priority_uld_cnt = None
  with open(args.output) as f:
    cost, used_pkg, priority_uld_cnt = [int(x) for x in f.readline().split(',')]
    for i in range(m):
      vals = f.readline().split(',')
      pkg_id = int(vals[0].split('-')[-1])
      if vals[1] == 'NONE':
        assert(all([int(x) == -1 for x in vals[2:]]))
        continue

      uld_id = int(vals[1][1:])
      x_min, y_min, z_min, x_max, y_max, z_max = (int(x) for x in vals[2:])
      pkg_positions[pkg_id] = PkgPosition(uld_id, x_min, y_min, z_min, x_max, y_max, z_max)

  calculated_cost = 0
  priority_uld_ids = set()
  weights = [0 for i in range(n + 1)]
  
  for i in range(1, m + 1):
    pkg = packages[i]
    pkgp = pkg_positions[i]
    if pkgp == None:
      if pkg.priority == 'P':
        raise ConstraintException('Priority package not used!')
      calculated_cost += pkg.cost
      continue
    
    uld_id = pkg_positions[i].uld_id
    uld = ulds[uld_id]
    
    used_pkg -= 1
    if pkg.priority == 'P':
      priority_uld_ids.add(uld_id)

    dims = [pkgp.x_max - pkgp.x_min, pkgp.y_max - pkgp.y_min, pkgp.z_max - pkgp.z_min]
    dims1 = sorted([pkg.length, pkg.width, pkg.height])
    dims2 = sorted(dims)
    if any([dims1[i] != dims2[i] for i in range(3)]):
      raise ConstraintException(f'Dimensions of P-{i} do not match!')
    
    check_x = pkgp.x_min < 0 or pkgp.x_max > uld.length
    check_y = pkgp.y_min < 0 or pkgp.y_max > uld.width
    check_z = pkgp.z_min < 0 or pkgp.z_max > uld.height
    if check_x or check_y or check_z:
      raise ConstraintException(f'Dimensions of P-{i} exceed dimensions of ULD-{uld_id}!')
    
    weights[uld_id] += pkg.weight

    # --- UPDATED FLOATING LOGIC ---
    # A package is supported if:
    # 1. It is on the ground (z_min == 0)
    # 2. OR it has ANY overlap (> 0) with a package directly below it
    
    is_supported = (pkgp.z_min == 0) # True if on ground
    
    for j in range(1, m + 1):
      if i == j or pkg_positions[j] == None or uld_id != pkg_positions[j].uld_id:
        continue

      if check_intersection(pkgp, pkg_positions[j]):
        raise ConstraintException(f'P-{i} and P-{j} intersect!')
      
      # If we haven't found support yet, check if this neighbor supports it
      if not is_supported:
        if pkg_positions[j].z_max == pkgp.z_min:
          if get_xy_overlap_area(pkgp, pkg_positions[j]) > 0:
            is_supported = True
            # We don't break here because we still need to check intersection 
            # with ALL other packages to ensure validity. 
            # If performance is an issue, we can separate the intersection loop 
            # and the support loop.

    if not is_supported:
      raise ConstraintException(f'P-{i} is floating! (z_min={pkgp.z_min} but no package below it)')
    # ------------------------------
  
  if used_pkg != 0:
    raise ConstraintException('Number of used packages do not match!')
  
  if len(priority_uld_ids) != priority_uld_cnt:
    raise ConstraintException('Number of priority ULDs do not match!')
  
  for i in range(1, n + 1):
    if weights[i] > ulds[i].weight_limit:
      raise ConstraintException(f'Weight limit of ULD-{i} exceeded!')
  
  calculated_cost += priority_uld_cnt * k
  if cost != calculated_cost:
    raise ConstraintException(f'Cost should be {calculated_cost}!')
    
  print('All constraints satisfied!')

if __name__ == '__main__':
  try:
    main()
  except ConstraintException as e:
    print(e.args[0])