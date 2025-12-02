import json
import sys

def generate_report(json_file, output_file):
    """
    Generate a results report from training JSON
    Format:
    cost,total_packages_packed,boxes_with_priority
    P-ID,box_id,x1,y1,z1,x2,y2,z2
    """
    
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    report_lines = []
    
    # Process each iteration
    for iteration in data.get('iterations', []):
        iteration_num = iteration['iteration']
        report_lines.append(f"\n{'='*60}")
        report_lines.append(f"ITERATION {iteration_num}")
        report_lines.append(f"{'='*60}\n")
        
        # Game generation statistics
        game_gen = iteration.get('game_generation', {})
        stats = game_gen.get('statistics', {})
        
        avg_baseline = stats.get('avg_baseline_reward', 0)
        avg_mcts = stats.get('avg_mcts_reward', 0)
        std_mcts = stats.get('std_mcts_reward', 0)
        games_improved = stats.get('games_improved', 0)
        total_games = stats.get('games_total', 1)
        
        report_lines.append(f"GAME GENERATION RESULTS:")
        report_lines.append(f"  Average Baseline Reward: {avg_baseline:.4f}")
        report_lines.append(f"  Average MCTS Reward: {avg_mcts:.4f} ± {std_mcts:.4f}")
        report_lines.append(f"  Games Improved: {games_improved}/{total_games}")
        report_lines.append(f"  Improvement Rate: {(games_improved/total_games)*100:.1f}%\n")
        
        # Per-game details
        for game in game_gen.get('games', []):
            game_idx = game['game_index']
            baseline = game.get('baseline_reward', 0)
            mcts = game.get('mcts_reward', 0)
            improvement = game.get('improvement', 0)
            num_packages = len(game.get('initial_packages', []))
            num_moves = len(game.get('moves', []))
            
            report_lines.append(f"Game {game_idx}:")
            report_lines.append(f"  Packages: {num_packages}")
            report_lines.append(f"  Container Height: {game.get('container_height', 24)}")
            report_lines.append(f"  Moves Taken: {num_moves}")
            report_lines.append(f"  Baseline Reward: {baseline:.4f}")
            report_lines.append(f"  MCTS Reward: {mcts:.4f}")
            report_lines.append(f"  Improvement: {improvement:.4f} ({(improvement/baseline)*100:.1f}%)")
            
            # Top 5 moves for each step
            moves = game.get('moves', [])
            if moves:
                report_lines.append(f"  First Move Top Actions:")
                first_move = moves[0]
                for action in first_move.get('top_5_actions', [])[:3]:
                    action_id = action['action_id']
                    prob = action['probability']
                    report_lines.append(f"    Action {action_id}: {prob:.4f}")
            
            report_lines.append("")
        
        # Training results
        training = iteration.get('training', {})
        if 'history' in training:
            report_lines.append(f"TRAINING RESULTS:")
            report_lines.append(f"  Data Points: {training.get('data_points', 0)}")
            report_lines.append(f"  Augmented Points: {training.get('augmented_points', 0)}")
            
            for hist in training['history']:
                if 'epoch' in hist:
                    report_lines.append(f"    Epoch {hist['epoch']}: Loss = {hist['train_loss']:.4f}")
                elif 'test_loss' in hist:
                    report_lines.append(f"    Test Loss: {hist['test_loss']:.4f}")
            
            report_lines.append("")
    
    # Summary statistics across all iterations
    report_lines.append(f"\n{'='*60}")
    report_lines.append("OVERALL TRAINING SUMMARY")
    report_lines.append(f"{'='*60}\n")
    
    all_baseline = []
    all_mcts = []
    
    for iteration in data.get('iterations', []):
        stats = iteration.get('game_generation', {}).get('statistics', {})
        all_baseline.append(stats.get('avg_baseline_reward', 0))
        all_mcts.append(stats.get('avg_mcts_reward', 0))
    
    if all_baseline and all_mcts:
        import statistics
        report_lines.append(f"Iterations Completed: {len(all_baseline)}")
        report_lines.append(f"Average Baseline Reward (all iterations): {statistics.mean(all_baseline):.4f}")
        report_lines.append(f"Average MCTS Reward (all iterations): {statistics.mean(all_mcts):.4f}")
        report_lines.append(f"Overall Improvement: {(statistics.mean(all_mcts) - statistics.mean(all_baseline)):.4f}")
    
    report_lines.append(f"\nDevice Used: {data.get('device', 'Unknown')}")
    report_lines.append(f"Training Started: {data.get('training_start', 'Unknown')}")
    
    # Write to file
    with open(output_file, 'w') as f:
        f.write('\n'.join(report_lines))
    
    # Print to console
    print('\n'.join(report_lines))
    print(f"\n✓ Report saved to: {output_file}")

if __name__ == '__main__':
    json_file = 'training_results.json'
    output_file = 'training_report.txt'
    
    if len(sys.argv) > 1:
        json_file = sys.argv[1]
    if len(sys.argv) > 2:
        output_file = sys.argv[2]
    
    generate_report(json_file, output_file)
