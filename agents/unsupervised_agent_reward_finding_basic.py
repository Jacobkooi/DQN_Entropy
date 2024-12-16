
from networks import EncoderDMC, DQNmodel_shallow, EncoderDMC_lowdim
from utils import to_numpy
from replaybuffer import ReplayBuffer
import torch
import torch.nn.functional as F
import numpy as np
import random as r
import wandb
import matplotlib.pyplot as plt
import torch.nn as nn
import socket
import os


class Agent_Reward_Finding_Basic:

    def __init__(self, env, eval_env,args=None):

        # Create a tanh initialization
        def weights_init(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=args.gain)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        if torch.cuda.is_available():
            self.device = 'cuda'
        else:
            self.device = 'cpu'

        self.name = 'fourmaze_3states'
        self.env = env
        self.eval_env = eval_env
        self.activation = args.activation
        self.gain = args.gain
        self.hostname = socket.gethostname()

        # Gather every state in every environment, for plotting purposes
        self.maze_size = env._size_maze
        self.higher_dim_obs = env._higher_dim_obs
        self.latent_dim = args.latent_dim
        self.batch_size = args.batch_size
        self.eps = args.eps_start
        self.eps_end = args.eps
        self.tau = args.tau
        self.gamma = args.gamma
        self.reward_scaler = args.reward_scaler
        self.map_type = args.map_type

        self.batch_entropy_scaler = args.batch_entropy_scaler
        self.dqn_architecture = args.dqn_architecture
        self.subsequent = args.subsequent

        self.lr = args.lr_encoder

        self.lr_dqn = args.lr_dqn

        self.onehot = True
        self.action_dim = 4

        self.dqn_scaler = args.dqn_scaler
        self.prediction_delta = True

        self.output = dict()
        self.iterations = 0

        # Convolutional encoder
        if self.higher_dim_obs:
            self.encoder = EncoderDMC(latent_dim=self.latent_dim, maze_size=self.maze_size, activation=self.activation).to(self.device)
            if self.activation == 'tanh+initialization':
                self.encoder.apply(weights_init)
            self.encoder = torch.jit.trace(self.encoder,
                                           (torch.rand(1, 1, self.maze_size * 6, self.maze_size * 6).to(self.device),))

        else:
            self.encoder = EncoderDMC_lowdim(latent_dim=self.latent_dim, maze_size=self.maze_size).to(self.device)
            self.encoder = torch.jit.trace(self.encoder, (torch.rand(1, 1, self.maze_size, self.maze_size).to(self.device),))

        # DQN network
        #     torch.set_float32_matmul_precision('high')
        self.dqn = torch.compile(DQNmodel_shallow(input_dim=self.latent_dim, depth=1).to(self.device))
        self.target_dqn = torch.compile(DQNmodel_shallow(input_dim=self.latent_dim, depth=1).to(self.device))
        self.target_dqn.load_state_dict(self.dqn.state_dict())

        # Optimizer encoder
        self.encoder_optimizer = torch.optim.Adam(self.encoder.parameters(), lr=self.lr, )

        # Optimizer DQN
        self.dqn_optimizer = torch.optim.Adam(self.dqn.parameters(), lr=self.lr_dqn)

        # Replay Buffer
        self.buffer = ReplayBuffer(np.expand_dims(self.env.observe()[0], axis=0).shape, env.action_space.shape[0], int(args.fill_buffer)+int(args.iterations)+50000, self.device)

        # DQN loss torch.jit.trace
        self.dqn_loss = torch.jit.trace(self.compute_DQN_Loss,
                                        (torch.rand(1, self.action_dim).to(self.device),
                                         torch.rand(1, self.action_dim).to(self.device),
                                         torch.rand(1, self.action_dim).to(self.device),
                                         torch.rand(1, 1).to(self.device),
                                         torch.rand(1, 1).to(self.device)))

        self.entropy_loss = torch.jit.trace(self.compute_entropy_loss, (torch.rand(1, self.latent_dim).to(self.device)))
        self.entropy_loss_subs = torch.jit.trace(self.compute_entropy_loss_subs,
                                                 (torch.rand(1, self.latent_dim).to(self.device),
                                                  torch.rand(1, self.latent_dim).to(self.device)))

        self.best_reward = -999
        self.average_reward = 0
        self.q_loss = 0
        self.consecutive_distance = 0
        self.batch_distance = 0
        self.feature_consecutive_distance = 0
        self.feature_batch_distance = 0
        self.fourth_layer_q_latent_batch_distance = 0
        self.fourth_layer_q_latent_consecutive_distance = 0
        self.last_layer_q_latent_batch_distance = 0
        self.last_layer_q_latent_consecutive_distance = 0
        self.consecutive_q_distance = 0
        self.mean_q_value = 0
        self.intersample_q_distance = 0

        self.moving_average = 0.97

    def mlp_learn(self):

        STATE, ACTION, REWARD, NEXT_STATE, DONE = self.buffer.sample(self.batch_size)

        # speed up gradient removal:
        for param in self.encoder.parameters():
            param.grad = None
        for param in self.dqn.parameters():
            param.grad = None

        # One-hot action encodings
        if self.onehot:
            ACTION = F.one_hot(ACTION.squeeze(1).long(), num_classes=self.action_dim)

        # Current latents (Z_t)
        full_latent, pre_activation = self.encoder(STATE)
        # Find dying relu activations and create a loss that makes them slightly above 0

        # Next latents (Z_t+1)
        next_latent, next_pre_activation = self.encoder(NEXT_STATE)

        Q = self.dqn(full_latent)
        # next_Q = self.dqn(next_latent)
        target_Q = self.target_dqn(next_latent)
        q_loss = self.dqn_loss(Q, target_Q, ACTION, REWARD*self.reward_scaler, DONE)
        loss = q_loss

        # Plot the latents
        if self.iterations % 10000 == 0 and self.hostname == 'jacobk-Legion-7-16ITHg6':
            for j in range(9):
                plt.subplot(3, 3, j + 1)
                for i in range(self.batch_size):
                    plt.plot(full_latent[i][1 + 50 * j].detach().cpu().numpy(),
                             full_latent[i][2 + 50 * j].detach().cpu().numpy(), 'o')
                if self.activation == 'tanh' or self.activation == 'tanh+initialization' or self.activation == 'layernorm' or self.activation == 'linear':
                    plt.axis([-1, 1, -1, 1])
                    if j==0 or j==3:
                        # hide x axis
                        plt.setp(plt.gca(), 'xticklabels', [])
                    elif j==6:
                        continue
                    elif j==7 or j==8:
                        # hide y axis
                        plt.setp(plt.gca(), 'yticklabels', [])
                    else:
                        # hide all axis:
                        plt.setp(plt.gca(), 'xticklabels', [])
                        plt.setp(plt.gca(), 'yticklabels', [])
                elif self.activation == 'layernorm':
                    plt.axis([-1, 1, -1, 1])
                    if j==0 or j==3:
                        # hide x axis
                        plt.setp(plt.gca(), 'xticklabels', [])
                    elif j==6:
                        continue
                    elif j==7 or j==8:
                        # hide y axis
                        plt.setp(plt.gca(), 'yticklabels', [])
                    else:
                        # hide all axis:
                        plt.setp(plt.gca(), 'xticklabels', [])
                        plt.setp(plt.gca(), 'yticklabels', [])

            # Create images directory if it doesnt yet exist:
            # Create images directory if it doesn't yet exist
            if not os.path.exists('images'):
                os.makedirs('images')

            file_name = f'images/plot_{self.iterations}.png'
            plt.subplots_adjust(hspace=0.2) # 0.53
            plt.subplots_adjust(wspace=0.2) # 0.5
            # Save the plot as a PDF file
            plt.savefig(file_name, format='png')
            plt.close()
            plt.cla()
            plt.clf()

        if self.batch_entropy_scaler !=50:
            if self.subsequent:
                loss_batch_entropy = self.compute_entropy_loss_subs(full_latent, next_latent)
            else:
                loss_batch_entropy = self.compute_entropy_loss(full_latent)
            loss += loss_batch_entropy

        # Log all the losses with wandb
        self.q_loss = self.moving_average * self.q_loss + (1-self.moving_average) * q_loss.item()

        if self.iterations % 1000 == 0:
            with torch.no_grad():
                mean_q_value = torch.mean(Q)
                # calculate mean and std of all the network weights

                # calculate mean of latent state, so 1 single mean and 1 single std over both batch dimension and neuron dimension.
                mean_latent = torch.mean(full_latent)
                std_latent = torch.std(full_latent)
                difference_states1 = full_latent - torch.roll(full_latent, 1, dims=0)
                latent_entropy = torch.norm(difference_states1, dim=1, p=2).mean()

            self.mean_q_value = self.moving_average * self.mean_q_value + (1 - self.moving_average) * mean_q_value

            wandb.log({
                       "mean_Q_value": self.mean_q_value,
                        "latent_entropy": latent_entropy,
                          "q_loss": self.q_loss,
                            "mean_latent": mean_latent,
                            "std_latent": std_latent,
                            "entropy_iterations": self.iterations})
        # Backprop the loss
        loss.backward()

        self.encoder_optimizer.step()
        self.dqn_optimizer.step()

        # target network update
        for target_param, param in zip(self.target_dqn.parameters(), self.dqn.parameters()):
            target_param.data.copy_(self.tau * param + (1 - self.tau) * target_param)

        # Print the losses and predictions every 500 iterations
        if self.iterations % 500 == 0:
            print("Iterations", self.iterations)

        self.iterations += 1

    def get_action(self, latent):

        if np.random.rand() < self.eps:
            return self.env.actions[r.randrange(4)]
        else:
            with torch.no_grad():
                q_vals = self.dqn(latent)
            action = np.argmax(to_numpy(q_vals))
            return action

    def run_agent(self):

        done = False
        state = self.env.observe()
        latent, _ = self.encoder(torch.as_tensor(state).unsqueeze(0).float().to(self.device))
        action = self.get_action(latent)
        reward = self.env.step(action)
        next_state = self.env.observe()

        if self.env.inTerminalState():
            self.env.reset(1)
            done = True
        self.buffer.add(state, action, reward, next_state, done)

        if self.iterations % 10000 == 0:
            if self.iterations ==0:
                self.evaluate(eval_episodes=100)
            else:
                self.evaluate()

        self.mlp_learn()
        self.eps = max(self.eps_end, self.eps - 0.8/50000)

    def evaluate(self, eval_episodes=100):

        self.eval_env.reset(1)
        average_reward = []
        Average_reward = 0
        solved = []
        Solved = 0

        for i in range(eval_episodes):
            reward = []
            done=False
            while not done:

                state = self.eval_env.observe()
                latent, _ = self.encoder(torch.as_tensor(state).unsqueeze(0).float().to(self.device))

                action = self.get_action(latent)
                reward_t = self.eval_env.step(action, dont_take_reward=False)
                reward.append(reward_t)

                if self.eval_env.inTerminalState():
                    solved.append(self.eval_env.inTerminalState(solved=True)[1])
                    self.eval_env.reset(1)
                    done = True
                    reward = sum(reward)
                    average_reward.append(reward)

        # Calculate the percentage of solved episodes
        Solved += sum(solved)/len(solved)
        Average_reward += sum(average_reward)/len(average_reward)

        if Average_reward >= self.best_reward:
            self.best_reward = Average_reward
            wandb.log({'Best reward': self.best_reward,
                       'Solved': Solved})
        wandb.log({'average_reward': Average_reward,
                   'global_iterations': self.iterations})

        print('The AVERAGE REWARD is:', Average_reward)
        print('The PERCENTAGE OF SOLVED EPISODES is:', Solved*100, '%')

    def compute_DQN_Loss(self, Q, target_Q, actions, rewards, dones):
        # Change actions to long format for the gather function
        actions = actions.long()
        actions = torch.argmax(actions, dim=1).unsqueeze(1)
        # Current timestep Q-values with the actions from the minibatch
        Q = Q.gather(1, actions)

        # In DQN, we directly use the target network to evaluate the Q-value of the next state
        # for the action that maximizes the Q-value according to the target network itself.
        # Note: The difference here is we're using next_Q instead of target_Q to select the action.
        # However, since it's DQN, we should be using target_Q directly for consistency with the DQN algorithm.
        # The correct approach would be to use target_Q for both selecting and evaluating the action's Q-value.
        max_next_Q_values = target_Q.max(1)[0].unsqueeze(1)

        # Compute the target Q-value
        Q_target = rewards + (1 - dones.int()) * self.gamma * max_next_Q_values

        # Calculate the loss
        loss = F.mse_loss(Q, Q_target.detach())

        return loss

    def compute_entropy_loss(self, latent):

        latent_rolled = torch.roll(latent, 1, dims=0)
        difference_states = latent - latent_rolled

        # normal random states loss
        loss = torch.exp(-self.batch_entropy_scaler * torch.norm(difference_states, dim=1, p=2)).mean()

        return loss

    def compute_entropy_loss_subs(self, latent, next_latent):

        difference_states = latent - next_latent

        # normal random states loss
        loss = torch.exp(-self.batch_entropy_scaler * torch.norm(difference_states, dim=1, p=2)).mean()

        return loss