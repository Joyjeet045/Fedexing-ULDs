import sys
import os
import json
from datetime import datetime

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../src/container-solver'))
from package_utils import random_package, normalize_packages
from container_solver import Container, generate_episode, calculate_baseline_reward
from policy_value_network import PolicyValueNetwork, train_policy_value_network

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import requests
import pickle
import argparse
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if device.type != 'cuda':
    try:
        import torch_directml
        device = torch_directml.device()
    except ImportError:
        pass

class TrainingLogger:
    def __init__(self, output_file):
        self.output_file = output_file
        self.results = {
            "training_start": datetime.now().isoformat(),
            "device": str(device),
            "iterations": []
        }
    
    def log_iteration(self, iteration_data):
        self.results["iterations"].append(iteration_data)
        self.save()
    
    def save(self):
        self.results["training_end"] = datetime.now().isoformat()
        with open(self.output_file, 'w') as f:
            json.dump(self.results, f, indent=2)

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

def generate_training_data_with_logging(
    games_per_iteration, simulations_per_move, 
    c_puct, virtual_loss, thread_count, batch_size, 
    addresses, episodes_file):
    
    games_data = []
    baseline_rewards = []
    mcts_rewards = []
    
    for game_idx in tqdm(range(games_per_iteration), desc="Generating games"):
        container_height = 24
        packages = [random_package() for _ in range(Container.package_count)]
        container = Container(container_height, packages)
        
        game_info = {
            "game_index": game_idx,
            "container_height": container_height,
            "initial_packages": [
                {"index": i, "dimensions": [packages[i].shape.x, packages[i].shape.y, packages[i].shape.z]}
                for i in range(len(packages))
            ],
            "moves": []
        }
        
        baseline_reward = calculate_baseline_reward(container, addresses)
        baseline_rewards.append(baseline_reward)
        
        episode = generate_episode(
            container, simulations_per_move,
            c_puct, virtual_loss, thread_count,
            batch_size, addresses
        )
        
        for step_idx, state_eval in enumerate(episode):
            priors = state_eval.priors
            top_actions = sorted(enumerate(priors), key=lambda x: x[1], reverse=True)[:5]
            
            packages_remaining = len([p for p in state_eval.container.packages if not p.is_placed])
            
            move_info = {
                "step": step_idx,
                "packages_remaining": packages_remaining,
                "current_reward": float(state_eval.container.reward),
                "top_5_actions": [
                    {"action_id": idx, "probability": float(prob)}
                    for idx, prob in top_actions
                ]
            }
            game_info["moves"].append(move_info)
            
            height_map = np.array(state_eval.container.height_map, dtype=np.float32) / state_eval.container.height
            height_map = np.expand_dims(height_map, axis=0)
            packages_data = normalize_packages(state_eval.container)
            priors_array = np.array(priors, dtype=np.float32)
            reward_array = np.array([state_eval.reward], dtype=np.float32)
            pickle.dump((height_map, packages_data, priors_array, reward_array), episodes_file)
        
        mcts_reward = episode[-1].reward
        mcts_rewards.append(mcts_reward)
        
        game_info["baseline_reward"] = float(baseline_reward)
        game_info["mcts_reward"] = float(mcts_reward)
        game_info["improvement"] = float(mcts_reward - baseline_reward)
        games_data.append(game_info)
        
        relative_reward = +1 if mcts_reward > baseline_reward else -1
        for state_eval in episode:
            state_eval.reward = relative_reward
    
    baseline_rewards = np.array(baseline_rewards)
    mcts_rewards = np.array(mcts_rewards)
    
    summary = {
        "games": games_data,
        "statistics": {
            "avg_baseline_reward": float(baseline_rewards.mean()),
            "std_baseline_reward": float(baseline_rewards.std()),
            "avg_mcts_reward": float(mcts_rewards.mean()),
            "std_mcts_reward": float(mcts_rewards.std()),
            "games_improved": int(sum(1 for b, m in zip(baseline_rewards, mcts_rewards) if m > b)),
            "games_total": len(baseline_rewards)
        }
    }
    
    print(f'Average baseline reward: {baseline_rewards.mean():.3f} ± {baseline_rewards.std():.3f}')
    print(f'Average MCTS reward: {mcts_rewards.mean():.3f} ± {mcts_rewards.std():.3f}')
    
    return summary

def train_with_logging(model, trainloader, testloader, device, epochs=2):
    model.train()
    learning_rate = 0.005
    momentum = 0.9
    
    from torch import nn, optim
    criterion_policy = nn.CrossEntropyLoss()
    criterion_value = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=1e-4)
    
    training_history = []
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        batch_count = 0
        for inputs in trainloader:
            inputs = (tensor.to(device) for tensor in list(inputs))
            image_data, package_data, priors, reward = inputs
            
            predicted_priors, predicted_reward = model(image_data, package_data)
            loss = criterion_policy(predicted_priors, priors) + criterion_value(predicted_reward, reward)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            batch_count += 1
        
        avg_loss = epoch_loss / max(batch_count, 1)
        training_history.append({"epoch": epoch + 1, "train_loss": avg_loss})
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}')
    
    if testloader is not None and len(testloader) > 0:
        model.eval()
        with torch.no_grad():
            total_loss = 0.0
            batch_count = 0
            for inputs in testloader:
                inputs = (tensor.to(device) for tensor in list(inputs))
                image_data, package_data, priors, reward = inputs
                predicted_priors, predicted_reward = model(image_data, package_data)
                loss = criterion_policy(predicted_priors, priors) + criterion_value(predicted_reward, reward)
                total_loss += loss.item()
                batch_count += 1
            if batch_count > 0:
                avg_test_loss = total_loss / batch_count
                training_history.append({"test_loss": avg_test_loss})
                print(f'Test Loss: {avg_test_loss:.4f}')
    
    return training_history

def perform_iteration_with_logging(
    iteration_num, model_path, addresses, episodes_file_path,
    games_per_iteration, simulations_per_move, epochs):
    
    iteration_data = {
        "iteration": iteration_num,
        "timestamp": datetime.now().isoformat(),
        "parameters": {
            "games_per_iteration": games_per_iteration,
            "simulations_per_move": simulations_per_move,
            "epochs": epochs
        }
    }
    
    if not os.path.exists(model_path):
        policy_value_network = PolicyValueNetwork()
        torch.save(policy_value_network.state_dict(), model_path)
        print("Created new model")
    
    print('UPLOADING MODEL...')
    with open(model_path, 'rb') as model:
        for address in addresses:
            model.seek(0)
            files = {'file': model}
            response = requests.post('http://' + address + '/policy_value_upload', files=files)
            if response.text == 'success':
                print(f'Model uploaded to {address}')
            else:
                raise Exception(f'Upload failed: {address}')
    
    print('\nGENERATING GAMES:')
    with open(episodes_file_path, 'w'):
        pass
    
    with open(episodes_file_path, 'ab') as file:
        game_summary = generate_training_data_with_logging(
            games_per_iteration, simulations_per_move,
            5.0, 3, 8, 4, addresses, file
        )
    
    iteration_data["game_generation"] = game_summary
    
    print('\nTRAINING:')
    train_data, test_data = get_test_train_data(episodes_file_path, ratio=0.8)
    
    if len(train_data) == 0:
        print("No training data generated!")
        iteration_data["training"] = {"error": "No data"}
        return iteration_data
    
    train_dataset = ExperienceReplay(train_data)
    test_dataset = ExperienceReplay(test_data) if test_data else None
    
    trainloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    testloader = DataLoader(test_dataset, batch_size=32, shuffle=True) if test_dataset else None
    
    model = PolicyValueNetwork().to(device)
    model.load_state_dict(torch.load(model_path, weights_only=False))
    
    training_history = train_with_logging(model, trainloader, testloader, device, epochs)
    iteration_data["training"] = {
        "data_points": len(train_data),
        "augmented_points": len(train_dataset),
        "history": training_history
    }
    
    torch.save(model.state_dict(), model_path)
    print(f'Model saved to {model_path}\n')
    
    return iteration_data

def main():
    parser = argparse.ArgumentParser(description='Train RL container packing with detailed logging')
    parser.add_argument('--iterations', type=int, default=5, help='Number of training iterations')
    parser.add_argument('--games', type=int, default=4, help='Games per iteration')
    parser.add_argument('--simulations', type=int, default=32, help='MCTS simulations per move')
    parser.add_argument('--epochs', type=int, default=5, help='Training epochs per iteration')
    parser.add_argument('--model_path', default='policy_value_network.pth')
    parser.add_argument('--worker', default='127.0.0.1:8000')
    parser.add_argument('--output', default='training_results.json', help='Output results file')
    args = parser.parse_args()
    
    print("=" * 60)
    print("RL CONTAINER PACKING TRAINING")
    print("=" * 60)
    print(f"Iterations: {args.iterations}")
    print(f"Games per iteration: {args.games}")
    print(f"MCTS simulations per move: {args.simulations}")
    print(f"Training epochs per iteration: {args.epochs}")
    print(f"Device: {device}")
    print(f"Output file: {args.output}")
    print("=" * 60 + "\n")
    
    logger = TrainingLogger(args.output)
    logger.results["config"] = vars(args)
    
    addresses = [args.worker]
    
    for i in range(args.iterations):
        print(f"\n{'='*60}")
        print(f"ITERATION [{i+1}/{args.iterations}]")
        print(f"{'='*60}\n")
        
        iteration_data = perform_iteration_with_logging(
            i + 1, args.model_path, addresses, 'episodes.bin',
            args.games, args.simulations, args.epochs
        )
        
        logger.log_iteration(iteration_data)
        print(f"Results saved to {args.output}")
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE!")
    print(f"Results saved to: {args.output}")
    print(f"Model saved to: {args.model_path}")
    print("=" * 60)

if __name__ == '__main__':
    main()
