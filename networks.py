import torch
import torch.nn.functional as F
import torch.nn as nn


class EncoderDMC(nn.Module):
    """Convolutional encoder_C for image-based observations."""
    def __init__(self, latent_dim, scale=1, maze_size=8, activation=None):
        super().__init__()

        self.latent_dim = latent_dim
        self.obs_channels = 1
        self.scale = scale
        self.maze_size = maze_size
        # self.activation = nn.Tanh() if activation == 'tanh' else nn.ReLU() if activation == 'relu' else nn.PReLU(self.latent_dim) if \
        #     activation == 'prelu' else None
        if activation is None:
            raise NotImplementedError
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'tanh+initialization':
            self.activation = nn.Tanh()
        elif activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'leakyrelu':
            self.activation = nn.LeakyReLU()
        elif activation == 'prelu':
            self.activation = nn.PReLU(self.latent_dim)
        elif activation == 'layernorm_only':
            self.activation = nn.LayerNorm(self.latent_dim)
        elif activation == 'layernorm':
            self.activation = nn.Sequential(nn.LayerNorm(self.latent_dim),
                                            nn.Tanh())
        elif activation == 'linear':
            self.activation = nn.Identity()

        self.convs = nn.Sequential(
            nn.Conv2d(self.obs_channels, out_channels=32, kernel_size=(3, 3), stride=(2, 2)),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), stride=(1, 1)),
            nn.ReLU())

        if self.maze_size == 8:
            self.mlp = nn.Sequential(
                nn.Linear(in_features=14112, out_features=self.latent_dim),
                self.activation)
        elif self.maze_size == 10:
            self.mlp = nn.Sequential(
                nn.Linear(in_features=23328, out_features=self.latent_dim),      # 10
                self.activation)
        elif self.maze_size == 12:
            self.mlp = nn.Sequential(
                nn.Linear(in_features=34848, out_features=self.latent_dim),        # 12
                self.activation)
        elif self.maze_size == 14:
            self.mlp = nn.Sequential(
                nn.Linear(in_features=48672, out_features=self.latent_dim),  # 12
                self.activation)
        elif self.maze_size == 16:
            self.mlp = nn.Sequential(
                nn.Linear(in_features=83232, out_features=self.latent_dim),  # 12
                self.activation)

    def forward(self, obs):

        features = self.convs(obs)

        features = features.flatten(1)

        latent = self.mlp(features)
        pre_activation_latent = self.mlp[0](features)

        return latent, pre_activation_latent


class EncoderDMC_lowdim(nn.Module):
    """Convolutional encoder_C for image-based observations."""
    def __init__(self, latent_dim, scale=1, maze_size=8, activation=None):
        super().__init__()

        self.latent_dim = latent_dim
        self.obs_channels = 1
        self.scale = scale
        self.maze_size = maze_size
        if activation is None:
            raise NotImplementedError
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'relu':
            self.activation = nn.ReLU()

        if self.maze_size == 8:
            self.mlp = nn.Sequential(
                nn.Linear(in_features=self.maze_size**2, out_features=128),
                nn.ReLU(),
                nn.Linear(in_features=128, out_features=self.latent_dim),
                self.activation)

    def forward(self, obs):

        latent = self.mlp(obs.flatten(1))

        return latent, latent


class DQNmodel_shallow(nn.Module):
    """ Transition function MLP head for both w/o and w/ action"""
    def __init__(self, input_dim, depth=1, prediction_dim=4):
        super().__init__()

        if depth == 1 :
            self.linear_layers = nn.Sequential(
                nn.Linear(in_features=input_dim, out_features=prediction_dim),
            )
        elif depth == 2 :
            self.linear_layers = nn.Sequential(
                nn.Linear(in_features=input_dim, out_features=int(input_dim/2)),
                nn.Tanh(),
                nn.Linear(in_features=int(input_dim/2), out_features=prediction_dim),
            )
        elif depth == 3 :
            self.linear_layers = nn.Sequential(
                nn.Linear(in_features=input_dim, out_features=int(input_dim/2)),
                nn.Tanh(),
                nn.Linear(in_features=int(input_dim/2), out_features=int(input_dim/4)),
                nn.Tanh(),
                nn.Linear(in_features=int(input_dim/4), out_features=prediction_dim),
            )

    def forward(self, input):

        q_values = self.linear_layers(input)

        return q_values
