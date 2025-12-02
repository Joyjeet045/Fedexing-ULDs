from container_solver import Container, generate_episode, calculate_baseline_reward
from policy_value_network import PolicyValueNetwork, train_policy_value_network
from package_utils import random_package, normalize_packages

import os
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if device != 'cuda':
  try:
    import torch_directml
    device = torch_directml.device()
  except ImportError:
    pass

import requests
import pickle
import argparse
from tqdm import tqdm

def get_test_train_data(episodes_file_path, *, ratio):
  evaluations = []
  with open(episodes_file_path, 'rb') as f:
    while True:
      try:
        evaluations.append(pickle.load(f))
      except EOFError:
        break
  
  train_count = int(len(evaluations) * ratio)
  return evaluations[:train_count], evaluations[train_count:]

class ExperienceReplay(Dataset):
  def __init__(self, evaluations):
    self.evaluations = self.__augment_data(evaluations)

  def __augment_data(self, evaluations):
    augmented_data = []
    for image_data, package_data, priors, reward in evaluations:
      image_data = image_data[0]
      reflected = np.flip(image_data, axis=0)
      symmetries = (
        image_data,
        np.rot90(image_data, k=1),
        np.rot90(image_data, k=2),
        np.rot90(image_data, k=-1),
        reflected,
        np.rot90(reflected, k=1),
        np.rot90(reflected, k=2),
        np.rot90(reflected, k=-1)
      )

      for image_data in symmetries:
        image_data = np.expand_dims(image_data, axis=0)
        augmented_data.append((image_data.copy(), package_data, priors, reward))

    return augmented_data

  def __len__(self):
    return len(self.evaluations)
  
  def __getitem__(self, idx):
    return self.evaluations[idx]

def save_state_evaluation(evaluation, episodes_file):
  container = evaluation.container
  priors = evaluation.priors
  reward = evaluation.reward

  height_map = np.array(container.height_map, dtype=np.float32) / container.height
  height_map = np.expand_dims(height_map, axis=0)
  packages_data = normalize_packages(container)
  priors = np.array(priors, dtype=np.float32)
  reward = np.array([reward], dtype=np.float32)
  pickle.dump((height_map, packages_data, priors, reward), episodes_file)

def generate_training_data(
    games_per_iteration, simulations_per_move, 
    c_puct, virtual_loss, thread_count, batch_size, 
    addresses, episodes_file):
  baseline_rewards = []
  mcts_rewards = []
  relative_rewards = [0, 0]

  data_points_count = 0
  for _ in tqdm(range(games_per_iteration)):
    container_height = 24
    packages = [random_package() for _ in range(Container.package_count)]
    container = Container(container_height, packages)

    baseline_reward = calculate_baseline_reward(container, addresses)
    baseline_rewards.append(baseline_reward)

    episode = generate_episode(
      container, simulations_per_move, 
      c_puct, virtual_loss, thread_count, 
      batch_size, addresses
    )
    data_points_count += len(episode)

    mcts_reward = episode[-1].reward
    mcts_rewards.append(mcts_reward)

    relative_reward = (+1 if mcts_reward > baseline_reward else -1)
    relative_rewards[0 if relative_reward == -1 else 1] += 1

    for state_evaluation in episode:
      state_evaluation.reward = relative_reward
      save_state_evaluation(state_evaluation, episodes_file)

  print(f'{data_points_count} data points generated')

  baseline_rewards = np.array(baseline_rewards)
  mcts_rewards = np.array(mcts_rewards)
  print(f'Average baseline reward: {baseline_rewards.mean():.2} ± {baseline_rewards.std() * 100:.2}%')
  print(f'Average MCTS reward: {mcts_rewards.mean():.2} ± {mcts_rewards.std() * 100:.2}%')
  print(f'Relative Rewards -> (-{relative_rewards[0]}, +{relative_rewards[1]})')

def perform_iteration(model_path, addresses, episodes_file_path, generate_only=False):
  # Create model if it does not exist
  if not os.path.exists(model_path):
    policy_value_network = PolicyValueNetwork()
    torch.save(policy_value_network.state_dict(), model_path)

  # Uploaad model to all workers
  print('UPLOADING MODELS:')
  with open(model_path, 'rb') as model:
    for address in addresses:
      model.seek(0)
      files = { 'file': model }
      response = requests.post('http://' + address + '/policy_value_upload', files=files)
      if response.text == 'success':
        print(f'Model uploaded successfully to {address}')
      else:
        raise Exception(f'Model upload failed on worker: {address}')
  print()

  # Generate Games
  print('GENERATING GAMES:')
  with open(episodes_file_path, 'w'):
    pass

  with open(episodes_file_path, 'ab') as file:
    games_per_iteration = 8
    simulations_per_move = 64
    c_puct = 5.0
    virtual_loss = 3
    thread_count = 16
    batch_size = 4
    generate_training_data(
      games_per_iteration, simulations_per_move, 
      c_puct, virtual_loss, thread_count, batch_size, 
      addresses, file
    )

  print()

  if generate_only:
    return
  
  # Train
  print('TRAINING:')
  train_data, test_data = get_test_train_data(episodes_file_path, ratio=0.8)
  train_dataset = ExperienceReplay(train_data)
  test_dataset = ExperienceReplay(test_data)

  trainloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
  testloader = DataLoader(test_dataset, batch_size=32, shuffle=True)

  model = PolicyValueNetwork().to(device)
  model.load_state_dict(torch.load(model_path, weights_only=False))
  train_policy_value_network(model, trainloader, testloader, device)
  
  torch.save(model.state_dict(), model_path)
  print()

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--iteration_count', type=int, default=1)
  parser.add_argument('--model_path', default='policy_value_network.pth')
  parser.add_argument('--worker_addresses', default='127.0.0.1:8000')
  parser.add_argument('--generate_only', action='store_true')
  args = parser.parse_args()

  addresses = args.worker_addresses.split(',')
  for i in range(args.iteration_count):
    print(f'[{i + 1}/{args.iteration_count}]')
    perform_iteration(args.model_path, addresses, 'episodes.bin', args.generate_only)

if __name__ == '__main__':
  main()