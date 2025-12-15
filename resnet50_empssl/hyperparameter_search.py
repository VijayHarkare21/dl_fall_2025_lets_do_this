"""
EMP-SSL Hyperparameter Search Script

Performs grid search or random search over hyperparameters to find best configuration
for generalization (evaluated on CIFAR-10/CIFAR-100).

Usage:
    # Grid search
    python hyperparameter_search.py --mode grid --base_config configs/empssl_base.yaml
    
    # Random search (more efficient)
    python hyperparameter_search.py --mode random --n_trials 20 --base_config configs/empssl_base.yaml
"""

import os
import sys
import argparse
import yaml
import json
import time
from pathlib import Path
from datetime import datetime
from itertools import product
import random

import torch
import subprocess

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def load_base_config(config_path):
    """Load base configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def define_search_space():
    """
    Define hyperparameter search space
    
    Focus on parameters that affect generalization:
    - Learning rate (most important)
    - Lambda invariance weight
    - Epsilon squared
    - Batch size (affects training dynamics)
    - Optimizer (LARS, SGD, AdamW)
    """
    search_space = {
        'learning_rate': [0.01, 0.03, 0.1, 0.3],
        'lambda_inv': [100.0, 200.0, 400.0],
        'epsilon_sq': [0.1, 0.2, 0.5],
        'batch_size': [128, 256],  # Keep reasonable for memory
        'optimizer': ['lars', 'sgd', 'adamw'],  # Optimizer choice
    }
    return search_space


def generate_grid_search_configs(base_config, search_space):
    """Generate all combinations for grid search"""
    configs = []
    
    # Get all parameter names and their values
    param_names = list(search_space.keys())
    param_values = [search_space[name] for name in param_names]
    
    # Generate all combinations
    for combination in product(*param_values):
        config = base_config.copy()
        for name, value in zip(param_names, combination):
            config[name] = value
        
        # Create unique experiment name (handle strings and numbers)
        name_parts = []
        for n, v in zip(param_names, combination):
            if isinstance(v, str):
                name_parts.append(f'{n}_{v}')
            else:
                name_parts.append(f'{n}_{v}')
        config['experiment_name'] = f"empssl_hpsearch_{'_'.join(name_parts)}"
        configs.append(config)
    
    return configs


def generate_random_search_configs(base_config, search_space, n_trials=20):
    """Generate random combinations for random search"""
    configs = []
    
    for trial in range(n_trials):
        config = base_config.copy()
        
        # Randomly sample from each parameter
        for param_name, param_values in search_space.items():
            config[param_name] = random.choice(param_values)
        
        # Create unique experiment name
        config['experiment_name'] = f"empssl_hpsearch_random_{trial:03d}"
        configs.append(config)
    
    return configs


def train_model(config, output_dir):
    """
    Train model with given configuration
    
    Returns:
        checkpoint_path: Path to saved checkpoint
        training_time: Time taken for training
    """
    print(f"\n{'='*80}")
    print(f"Training with config:")
    print(f"  LR: {config['learning_rate']}")
    print(f"  Lambda: {config['lambda_inv']}")
    print(f"  Epsilon²: {config['epsilon_sq']}")
    print(f"  Batch size: {config['batch_size']}")
    print(f"{'='*80}\n")
    
    # Save temporary config file
    temp_config_path = output_dir / 'temp_config.yaml'
    with open(temp_config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    # Run training
    start_time = time.time()
    try:
        result = subprocess.run(
            [sys.executable, 'train_empssl.py', '--config', str(temp_config_path)],
            cwd=Path(__file__).parent,
            capture_output=True,
            text=True,
            timeout=3600  # 1 hour timeout per training run
        )
        
        if result.returncode != 0:
            print(f"ERROR: Training failed!")
            print(result.stderr)
            return None, None
        
        training_time = time.time() - start_time
        
        # Find checkpoint path (from experiment directory)
        # Experiment dir includes timestamp: {exp_name}_{timestamp}
        exp_name = config['experiment_name']
        exp_base_dir = Path(config.get('output_dir', './experiments'))
        checkpoint_path = None
        
        # Find the most recent experiment directory matching the name
        matching_dirs = list(exp_base_dir.glob(f"{exp_name}_*"))
        if matching_dirs:
            # Sort by modification time, get most recent
            most_recent = max(matching_dirs, key=lambda p: p.stat().st_mtime)
            checkpoint_dir = most_recent / 'checkpoints'
            if checkpoint_dir.exists():
                checkpoint_path = checkpoint_dir / 'checkpoint_epoch0.pth'
                if not checkpoint_path.exists():
                    checkpoint_path = checkpoint_dir / 'checkpoint_latest.pth'
        
        return checkpoint_path, training_time
        
    except subprocess.TimeoutExpired:
        print(f"ERROR: Training timed out after 1 hour")
        return None, None
    except Exception as e:
        print(f"ERROR: Training failed with exception: {e}")
        return None, None


def evaluate_model(checkpoint_path, eval_datasets=['cifar10']):
    """
    Evaluate model on specified datasets
    
    Returns:
        results: Dict with accuracy for each dataset
    """
    if checkpoint_path is None or not checkpoint_path.exists():
        return None
    
    results = {}
    
    for dataset in eval_datasets:
        print(f"\nEvaluating on {dataset}...")
        try:
            result = subprocess.run(
                [
                    sys.executable, 'eval_empssl.py',
                    '--checkpoint', str(checkpoint_path),
                    '--method', 'knn',
                    '--k', '20',
                    '--dataset', dataset,
                    '--batch_size', '256'
                ],
                cwd=Path(__file__).parent,
                capture_output=True,
                text=True,
                timeout=600  # 10 minute timeout per evaluation
            )
            
            if result.returncode == 0:
                # Parse accuracy from output
                output = result.stdout
                for line in output.split('\n'):
                    if f'k-NN (k=20) Accuracy:' in line:
                        acc_str = line.split('Accuracy:')[1].strip().replace('%', '')
                        results[dataset] = float(acc_str)
                        break
            else:
                print(f"ERROR: Evaluation on {dataset} failed")
                print(result.stderr)
                results[dataset] = None
                
        except Exception as e:
            print(f"ERROR: Evaluation on {dataset} failed with exception: {e}")
            results[dataset] = None
    
    return results


def main():
    parser = argparse.ArgumentParser(description='EMP-SSL Hyperparameter Search')
    parser.add_argument('--base_config', type=str, required=True,
                       help='Path to base configuration YAML file')
    parser.add_argument('--mode', type=str, choices=['grid', 'random'], default='random',
                       help='Search mode: grid or random')
    parser.add_argument('--n_trials', type=int, default=20,
                       help='Number of trials for random search')
    parser.add_argument('--eval_datasets', type=str, nargs='+', default=['cifar10'],
                       help='Datasets to evaluate on (e.g., cifar10 cifar100)')
    parser.add_argument('--output_dir', type=str, default='./hp_search_results',
                       help='Directory to save search results')
    
    args = parser.parse_args()
    
    # Load base config
    base_config = load_base_config(args.base_config)
    
    # Define search space
    search_space = define_search_space()
    
    # Generate configurations
    if args.mode == 'grid':
        configs = generate_grid_search_configs(base_config, search_space)
        print(f"Generated {len(configs)} configurations for grid search")
    else:
        configs = generate_random_search_configs(base_config, search_space, args.n_trials)
        print(f"Generated {len(configs)} configurations for random search")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save search configuration
    search_config = {
        'mode': args.mode,
        'n_trials': args.n_trials if args.mode == 'random' else len(configs),
        'search_space': search_space,
        'eval_datasets': args.eval_datasets,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    with open(output_dir / 'search_config.json', 'w') as f:
        json.dump(search_config, f, indent=2)
    
    # Run search
    all_results = []
    best_config = None
    best_score = -1.0
    
    print(f"\n{'='*80}")
    print(f"Starting hyperparameter search")
    print(f"Mode: {args.mode}")
    print(f"Total configurations: {len(configs)}")
    print(f"Evaluation datasets: {args.eval_datasets}")
    print(f"{'='*80}\n")
    
    start_time = time.time()
    
    for i, config in enumerate(configs):
        print(f"\n{'#'*80}")
        print(f"Configuration {i+1}/{len(configs)}")
        print(f"{'#'*80}")
        
        # Train
        checkpoint_path, training_time = train_model(config, output_dir)
        
        if checkpoint_path is None:
            print(f"SKIPPING: Training failed for config {i+1}")
            continue
        
        # Evaluate
        eval_results = evaluate_model(checkpoint_path, args.eval_datasets)
        
        if eval_results is None:
            print(f"SKIPPING: Evaluation failed for config {i+1}")
            continue
        
        # Calculate score (average across datasets, or just CIFAR-10)
        score = eval_results.get('cifar10', 0.0)
        if 'cifar100' in eval_results and eval_results['cifar100'] is not None:
            # Weighted average: 70% CIFAR-10, 30% CIFAR-100
            score = 0.7 * eval_results['cifar10'] + 0.3 * eval_results['cifar100']
        
        # Store results
        result = {
            'config_id': i,
            'config': config,
            'checkpoint_path': str(checkpoint_path),
            'training_time': training_time,
            'eval_results': eval_results,
            'score': score
        }
        all_results.append(result)
        
        # Update best
        if score > best_score:
            best_score = score
            best_config = config.copy()
            best_config['checkpoint_path'] = str(checkpoint_path)
            best_config['score'] = score
            best_config['eval_results'] = eval_results
        
        # Save intermediate results
        with open(output_dir / 'results.json', 'w') as f:
            json.dump(all_results, f, indent=2)
        
        # Print progress
        elapsed = time.time() - start_time
        avg_time = elapsed / (i + 1)
        remaining = avg_time * (len(configs) - i - 1)
        print(f"\nProgress: {i+1}/{len(configs)}")
        print(f"Current score: {score:.2f}%")
        print(f"Best score so far: {best_score:.2f}%")
        print(f"Elapsed: {elapsed/60:.1f} min, Estimated remaining: {remaining/60:.1f} min")
    
    # Final summary
    total_time = time.time() - start_time
    
    print(f"\n{'='*80}")
    print(f"HYPERPARAMETER SEARCH COMPLETE")
    print(f"{'='*80}")
    print(f"Total configurations tested: {len(all_results)}/{len(configs)}")
    print(f"Total time: {total_time/60:.1f} minutes")
    print(f"\nBest Configuration:")
    print(f"  Learning Rate: {best_config['learning_rate']}")
    print(f"  Lambda Inv: {best_config['lambda_inv']}")
    print(f"  Epsilon²: {best_config['epsilon_sq']}")
    print(f"  Batch Size: {best_config['batch_size']}")
    print(f"  Score: {best_score:.2f}%")
    print(f"  CIFAR-10: {best_config['eval_results'].get('cifar10', 'N/A'):.2f}%")
    if 'cifar100' in best_config['eval_results']:
        print(f"  CIFAR-100: {best_config['eval_results'].get('cifar100', 'N/A'):.2f}%")
    print(f"  Checkpoint: {best_config['checkpoint_path']}")
    print(f"{'='*80}\n")
    
    # Save final results
    final_results = {
        'search_config': search_config,
        'all_results': all_results,
        'best_config': best_config,
        'total_time': total_time
    }
    
    with open(output_dir / 'final_results.json', 'w') as f:
        json.dump(final_results, f, indent=2)
    
    # Save best config as YAML for easy reuse
    best_config_yaml = best_config.copy()
    best_config_yaml.pop('checkpoint_path', None)
    best_config_yaml.pop('score', None)
    best_config_yaml.pop('eval_results', None)
    
    with open(output_dir / 'best_config.yaml', 'w') as f:
        yaml.dump(best_config_yaml, f, default_flow_style=False)
    
    print(f"Results saved to: {output_dir}")
    print(f"Best config saved to: {output_dir / 'best_config.yaml'}")
    print(f"\nTo train final model with best config:")
    print(f"  python train_empssl.py --config {output_dir / 'best_config.yaml'}")


if __name__ == "__main__":
    main()