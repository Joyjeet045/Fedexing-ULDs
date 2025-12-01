# Genetic algorithm for the ULD box fitting problem
import random
from random import randint, choice, shuffle
from typing import List
import time
import pandas as pd
from utils import *
# seed=42
Genome = List[List[int]]
Population = List[Genome]
box = tuple[int, int, int, int, bool]

class ULD:
    def __init__(self, name:str, length: int, width: int, height: int, max_weight: int, boxes : List[tuple[int, int, int, int, int, int]] = None):
        self.name = name
        self.length = length
        self.width = width
        self.height = height
        self.max_weight = max_weight
        self.boxes = [] if boxes is None else boxes
        self.curweight = 0
        self.containp = False
    def overlaps(self, box: tuple[int, int, int, int, int, int]) -> bool:
        return any([overlap(box, placed_box) for placed_box in self.boxes])
    def add_box(self, box: box, pos: tuple[int, int, int]) -> None:
        # Add the box to the ULD
        box_loc = (pos[0], pos[1], pos[2], pos[0]+box[0], pos[1]+box[1], pos[2]+box[2])
        self.boxes.append(box_loc)
        self.curweight += box[3]
        if box[4]:
            self.containp = True

    def check_base(self, box: tuple[int, int, int, int, int, int], alpha=1) -> bool:
        # Check if the box is supported by the base of the ULD
        if box[2] == 0:
            return True
        support_boxes = [placed_box for placed_box in self.boxes if placed_box[5] == box[2]]
        # Check if the box is supported by any support box
        return any([support(placed_box, box) for placed_box in support_boxes])

    def can_fit(self, box: tuple[int, int, int, int, bool], pos: tuple[int, int, int]) -> bool:
        box_loc = (pos[0], pos[1], pos[2], pos[0]+box[0], pos[1]+box[1], pos[2]+box[2])
        if (pos[0]+box[0]>self.length or pos[1]+box[1]>self.width or pos[2]+box[2]>self.height):
            return False
        if (self.curweight+box[3]>self.max_weight):
            return False
        if self.overlaps(box_loc):
            return False
        if not self.check_base(box_loc):
            return False
        return True

        

class Box:
    def __init__(self, name: str, length: int, width: int, height: int, weight: int, priority: bool, cost: int = 100000):
        self.dimensions = [(length, width, height), (width, length, height),
                           (width, height, length), (height, width, length),
                           (height, length, width), (length, height, width)]
        self.name = name
        self.weight = weight
        self.priority = priority
        self.cost = cost

    def orient(self, orientation: int) -> tuple[int, int, int, int, bool]:
        return (*self.dimensions[orientation], self.weight, self.priority)


def generate_genome(length: int, seed: int = 42) -> Genome:
    # make a genome having a permutation of the numbers 0 to length-1 and a random integer from 0 to 5.
    # the permutation represents the order of the boxes to be placed in the ULDs.
    # the random integer is the orientation of the box.
    # set random seed for reproducibility
    #random.seed(seed)
    # Generating a random permutation of the numbers 0 to length-1
    genome = [[i, randint(0, 5)] for i in range(length)]
    shuffle(genome)
    return genome

    
# Function to find the correct position for insertion using binary search
def binary_insert(poset, new_pos):
    low, high = 0, len(poset)
    while low < high:
        mid = (low + high) // 2
        if poset[mid] < new_pos:
            low = mid + 1
        else:
            high = mid
    poset.insert(low, new_pos)

# Add multiple new positions to the sorted poset
def add_to_sorted(poset, new_positions):
    for pos in new_positions:
        binary_insert(poset, pos)

def generate_population(size: int, genome_length: int) -> Population:
    # generate a population of genomes
    return [generate_genome(genome_length) for _ in range(size)]



def fitness(genome: Genome, boxes: List[Box], uld_list: List[tuple], cost_type = "normal", k: int = 5000, min_dim:int = 0) -> int:
    # calculate the fitness of a genome
    assert cost_type in ["normal", "volume"], "Invalid cost type"
    unplaced = []
    placement = []
    poset = []
    ulds = [ULD(*uld) for uld in uld_list]
    for i in range(len(uld_list)):
        poset.append((0, 0, 0, i))
    min_box = Box("minbox", min_dim, min_dim, min_dim, 0, False)
    for box in range(len(genome)):
        box_id, orientation = genome[box]
        poset = (sorted(poset, key=lambda x: (x[3], x[0], x[2], x[1])))
        bad_pos = []
        for pos in poset:
            if not ulds[pos[3]].can_fit(min_box.orient(0), pos):
                bad_pos.append(pos)
        for pos in bad_pos:
            poset.remove(pos)
        for pos in poset:
            box_ = boxes[box_id].orient(orientation)
            l, w, h = box_[:3]
            res = ulds[pos[3]].can_fit(box_, pos)
            if res:
                ulds[pos[3]].add_box(box_, pos)
                placement.append((box_id, orientation, pos))
                poset.remove(pos)
                #add_to_sorted(poset,[(pos[0]+l, pos[1], pos[2], pos[3]), (pos[0], pos[1]+w, pos[2], pos[3]), (pos[0], pos[1], pos[2]+h, pos[3])])
                poset.extend([(pos[0]+l, pos[1], pos[2], pos[3]), (pos[0], pos[1]+w, pos[2], pos[3]), (pos[0], pos[1], pos[2]+h, pos[3])])
                break
            else:
                continue
        else:
            unplaced.append(box_id)
            placement.append((box_id, None))
    if cost_type == "normal":
        fitness = 0
        count = 0
        for uld in ulds:
            if uld.containp:
                fitness += k
                count += 1
        for box in unplaced:
            fitness += boxes[box].cost
        return (fitness, len(boxes)-len(unplaced), placement, count)
    elif cost_type == "volume":
        fitness = 0
        for box in unplaced:
            dims = boxes[box].orient(0)
            fitness += dims[0]*dims[1]*dims[2]
        return (fitness, len(boxes)-len(unplaced), placement)
    





def crossover(genome1: Genome, genome2: Genome) -> tuple[Genome]:
    # perform crossover between two genomes
    # choose two random indices to split the genomes
    # genome1 = [list(box) for box in genome1]
    # genome2 = [list(box) for box in genome2]
    if len(genome1) != len(genome2):
        print("ERROR: genomes of different lengths considered for crossover")
    n = len(genome1)
    # Check if there are duplicate boxes in the genome
    if len(set([box[0] for box in genome1])) != n or len(set([box[0] for box in genome2])) != n:
        print("ERROR: Duplicate boxes in the genome")
    i1, i2 = random.sample(range(len(genome1)), 2)
    split_index_1 = min(i1, i2)
    split_index_2 = max(i1, i2)
    if split_index_1 == 0 and split_index_2 == n-1:
        print("ERROR: whole genome considered for crossover")
    #print(split_index_1, split_index_2)
    c_1_mid = genome1[split_index_1:split_index_2+1]
    c_2_mid = genome2[split_index_1:split_index_2+1]
    child1 = [None for _ in range(n)]
    child2 = [None for _ in range(n)]
    child1[split_index_1:split_index_2+1] = c_1_mid
    child2[split_index_1:split_index_2+1] = c_2_mid
    #print(len(c_1_mid))
    def fill_remaining(child, parent, mid_segment):
        mid_perm = [box[0] for box in mid_segment]
        parent_index = 0
        for i in range(n):
            if child[i] is None:  # Find empty spots in the child
                while parent_index< len(parent) and parent[parent_index][0] in mid_perm:  # Skip values from the middle segment
                    parent_index += 1
                if parent_index < len(parent):
                    child[i] = parent[parent_index]
                    parent_index += 1
                else:
                    print(f"Error: parent_index out of bounds. mid_segment={mid_segment}, parent={parent}")
        return child
    child1 = fill_remaining(child1, genome2, c_1_mid)
    child2 = fill_remaining(child2, genome1, c_2_mid)
    return (child1, child2)

def mutate(genome: Genome, pm1:float, pm2:float, a: int = 0) -> Genome:
    # step 1: inversion of random segments:
    n = len(genome)
    genome = [list(box) for box in genome]
    if random.random() < pm1:
        i1, i2 = random.sample(range(a, len(genome)), 2)
        split_index_1 = min(i1, i2)
        split_index_2 = max(i1, i2)
        genome[split_index_1:split_index_2+1] = genome[split_index_1:split_index_2+1][::-1]
    # step 2: Change orientaions of boxes with a probability p2
    for i in range(a, n):
        if random.random() < pm2:
            genome[i][1] = randint(0, 5)
    return genome

def fitness_by_volume(genome: Genome, boxes: List[Box], ulds: List[tuple]):
    # calculate the fitness of a genome
    unplaced = []
    # placement = []
    poset = []
    
    uld = ULD(*uld)
    for box in range(len(genome)):
        box_id, orientation = genome[box]
        poset = (sorted(poset, key=lambda x: (x[0], x[2], x[1])))
        for pos in poset:
            res = uld.can_fit(boxes[box_id].orient(orientation), pos)
            if res:
                uld.add_box(boxes[box_id].orient(orientation))
                #placement.append((box_id, orientation, pos))
                poset.remove(pos)
                poset.append((pos[0]+boxes[box_id].orient(orientation)[0], pos[1], pos[2]))
                poset.append((pos[0], pos[1]+boxes[box_id].orient(orientation)[1], pos[2]))
                poset.append((pos[0], pos[1], pos[2]+boxes[box_id].orient(orientation)[2]))
                break
            else:
                continue
        else:
            unplaced.append(box_id)
            # placement.append((box_id, None))
    fitness = 0
    for box in unplaced:
        dims = boxes[box].orient(0)
        fitness += dims[0]*dims[1]*dims[2]
    return fitness

def check_duplicate(genome: Genome) -> bool:
    return len(set([box[0] for box in genome])) != len(genome)




def genetic_algorithm(ulds: List[tuple], boxes: List[Box], population_size: int, generations: int, pm1: float, pm2: float, initial_pop: list[Genome] = None, cost="volume", k:int = 5000, a:int=0) -> tuple[int, Genome]:
    # We use the elitist model for the genetic algorithm
    # set the initial best fitness to infinity
    assert cost in ["volume", "normal"]
    # Find minimum box dimension
    min_dim = min([min(box.orient(0)[:3]) for box in boxes])
    # print(min_dim)
    #min_dim = 0
    best_fitness = float('inf') 
    best_genome = None
    fit_list = []
    # generate the initial population
    population = generate_population(population_size, len(boxes)) if initial_pop is None else initial_pop
    #count = 0
    print(f"Population: {population_size}, Generations : {generations}, pm1 = {pm1}, pm2= {pm2}")
    for generation in range(generations):
        # calculate the fitness of each genome in the population
        fitnesses = [fitness(genome, boxes, ulds, cost_type=cost, k=k, min_dim=min_dim)[0] for genome in population]
        # find the genome with the best fitness
        min_fitness = min(fitnesses)
        # if min_fitness == 0:
        #     count+=1
        print(f'Generation {generation}: Min fitness: {min_fitness}')
        fit_list.append(min_fitness)
        if min_fitness < best_fitness:
            best_fitness = min_fitness
            best_genome = population[fitnesses.index(min_fitness)]
        # select the top 50% of the population having minimum fitnesses
        top_performers = [population[i] for i in np.argsort(fitnesses)][0:population_size//2]
        #print(min([fitness_by_volume(genome, boxes, uld) for genome in top_performers]))
        # create the next generation
        next_generation = []
        while len(next_generation)+len(top_performers) < population_size:
            # select two parents randomly from the top performers
            parent1, parent2 = random.sample(top_performers, 2)
            # perform crossover between the two parents
            child1, child2 = crossover(parent1, parent2)
            # mutate the children
            child1 = mutate(child1, pm1, pm2, a=a)
            child2 = mutate(child2, pm1, pm2, a=a)
            next_generation.extend([child1, child2])
        
        #print(min([fitness_by_volume(genome, boxes, uld) for genome in top_performers]))
        population = top_performers+next_generation
        # if count == 5:
        #     break
    print(f'Generation {generation+1}: Min fitness: {min_fitness}')
    return (population, best_genome, best_fitness, fit_list)


def ans_from_gene(uld_list, boxes, genome, cost_type='volume', k=5000):
    fit = fitness(genome, boxes, uld_list, cost_type=cost_type, k=k)
    placement = fit[2]
    ans = []
    for i in range(len(placement)):
        if placement[i][1] != None:
            l = boxes[placement[i][0]].orient(placement[i][1])[0]
            w = boxes[placement[i][0]].orient(placement[i][1])[1]
            h = boxes[placement[i][0]].orient(placement[i][1])[2]
            ans.append((boxes[placement[i][0]].name, uld_list[placement[i][2][3]][0], placement[i][2][0], placement[i][2][1], placement[i][2][2], placement[i][2][0]+l, placement[i][2][1]+w, placement[i][2][2]+h))
        else:
            ans.append((boxes[placement[i][0]].name, 'None', None, None, None, None, None, None))
    ans_df = pd.DataFrame(ans, columns=['Package Identifier', 'ULD Identifier', 'X0', 'Y0', 'Z0', 'X1', 'Y1', 'Z1'])
    ans_df = ans_df.sort_values(by='Package Identifier').reset_index(drop=True)
    return ans_df
