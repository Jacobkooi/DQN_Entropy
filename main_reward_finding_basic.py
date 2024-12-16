# Python script by Jacob Kooi
import os
from environments.maze_env import Maze
from agents.unsupervised_agent_reward_finding_basic import Agent_Reward_Finding_Basic
from utils import fill_buffer, set_seed
import numpy as np
import time
import argparse
import wandb

wandb.login()
os.environ["WANDB_SILENT"] = "true"

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_description', type=str, default='test_fourmaze')
    parser.add_argument('--GPU', type=str, default='0')
    parser.add_argument('--iterations', type=int, default=200000)
    parser.add_argument('--latent_dim', type=int, default=512)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--eps', type=float, default=0.1)
    parser.add_argument('--eps_start', type=float, default=0.1)
    parser.add_argument('--dqn_scaler', type=float, default=8)
    parser.add_argument('--lr_encoder', type=float, default=5e-5)
    parser.add_argument('--lr_dqn', type=float, default=5e-5)
    parser.add_argument('--maze_size', type=int, default=8)
    parser.add_argument('--format', type=str, default='png')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--gamma', type=float, default=0.85)
    parser.add_argument('--tau', type=float, default=0.02)
    parser.add_argument('--interval_iterations', type=int, default=10000)
    parser.add_argument('--fill_buffer', type=int, default=20000)
    parser.add_argument('--map_type', type=str, default='random_with_rewards')
    parser.add_argument('--batch_entropy_scaler', type=float, default=50)
    parser.add_argument('--dqn_architecture', type=str, default='shallow')
    parser.add_argument('--activation', type=str, default='tanh')
    parser.add_argument('--maze_rewards', type=int, default=3)
    parser.add_argument('--reward_scaler', type=float, default=1)
    parser.add_argument('--gain', type=float, default=5)
    parser.add_argument('--subsequent', type=int, default=0)

    args = parser.parse_args()
    wandb.init(
        project="Contrastive_DDQN",
            config=vars(args),
        )

    wandb.define_metric("global_iterations")
    wandb.define_metric("entropy_iterations")
    # define which metrics will be plotted against it
    wandb.define_metric("average_reward", step_metric="global_iterations")
    wandb.define_metric("q_loss", step_metric="entropy_iterations")
    wandb.define_metric("mean_latent", step_metric="entropy_iterations")
    wandb.define_metric("std_latent", step_metric="entropy_iterations")
    wandb.define_metric("latent_entropy", step_metric="entropy_iterations")

    set_seed(seed=args.seed)

rng = np.random.RandomState(123456)
# Maze params
higher_dim_bool = True

env = Maze(higher_dim_obs=higher_dim_bool, map_type=args.map_type, maze_size=args.maze_size, n_rewards=args.maze_rewards)
eval_env = Maze(higher_dim_obs=higher_dim_bool, map_type=args.map_type, maze_size=args.maze_size, n_rewards=args.maze_rewards)

env.create_map()
eval_env.create_map()

# Create the agent
agent = Agent_Reward_Finding_Basic(env, eval_env, args=args)

fill_buffer(agent.buffer, args.fill_buffer, env)

number_of_succesfull_episodes = np.where(np.array(agent.buffer.rewards) == 1)[0].shape[0]
percentage_chance_of_success = (number_of_succesfull_episodes / len(agent.buffer)) * 100
print("Number of rewards collected: ", number_of_succesfull_episodes, "Initial buffer size: ", len(agent.buffer))
print("Percentage chance of positive reward: ", percentage_chance_of_success)

# visualize_buffer_batch(agent)

# Make the directory specific for this environment and set of hyperparameters
iterations = []
rewards = []

# Main training loop
start_time = time.time()
intermediate_time = None
for i in range(args.iterations + 500):
    # Train the agent for an iteration
    agent.run_agent()
    if i % 5000 == 0:
        end_time = time.time()
        print("Iteration: ", i, "Time: ", end_time - start_time)
        start_time = time.time()



