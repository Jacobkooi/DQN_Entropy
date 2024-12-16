import torch


def compute_entropy_loss(latent, scaler):

    latent_rolled = torch.roll(latent, 1, dims=0)
    difference_states = latent-latent_rolled

    # normal random states loss
    loss = torch.exp(-scaler * torch.norm(difference_states, dim=1, p=2)).mean()

    return loss


def compute_entropy_loss_subs(latent, next_latent, scaler):

    difference_states = latent-next_latent

    # normal random states loss
    loss = torch.exp(-scaler * torch.norm(difference_states, dim=1, p=2)).mean()

    return loss
