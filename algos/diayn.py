import math
from collections import OrderedDict

import hydra
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


def weight_init(m):
    """Custom weight init for Conv2D and Linear layers."""
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        gain = nn.init.calculate_gain('relu')
        nn.init.orthogonal_(m.weight.data, gain)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)


class Discriminator(nn.Module):
    
    def __init__(self, obs_dim, skill_dim, hidden_dim):
        super().__init__()
        self.skill_pred_net = nn.Sequential(nn.Linear(obs_dim, hidden_dim),
                                            nn.ReLU(),
                                            nn.Linear(hidden_dim, hidden_dim),
                                            nn.ReLU(),
                                            nn.Linear(hidden_dim, skill_dim))
        self.apply(weight_init)

    def forward(self, obs):
        skill_pred = self.skill_pred_net(obs)
        return skill_pred


class DIAYN:
    """
    DIAYN w/PPO optimizer
    """
    def __init__(self,
                 actor_critic,
                 discriminator,
                 skill_dim,
                 update_encoder,
                 clip_param,
                 ppo_epoch,
                 num_mini_batch,
                 value_loss_coef,
                 entropy_coef,
                 lr=None,
                 eps=None,
                 max_grad_norm=None,
                 clip_value_loss=True,
                 log_grad_norm=False):

        # Actor critic model
        self.actor_critic = actor_critic

        # Discriminator model
        self.discriminator = discriminator
        self.skill_dim = skill_dim
        
        # PPO clip training params
        self.clip_param = clip_param
        self.ppo_epoch = ppo_epoch
        self.num_mini_batch = num_mini_batch
        self.clip_value_loss = clip_value_loss

        # PPO loss coefs
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef

        # PPO max gradient normalizer
        self.max_grad_norm = max_grad_norm

        # # Fix same LR for discriminator and PPO optimizer
        # self.lr = lr

        # Setup optimizer for PPO
        self.optimizer = optim.Adam(actor_critic.parameters(), lr=lr, eps=eps)

        # DIAYN discriminator functions
        self.discriminator_criterion = nn.CrossEntropyLoss()
        self.discriminator_opt = optim.Adam(self.discriminator.parameters(), lr=lr)
        self.discriminator.train()

        self.log_grad_norm = log_grad_norm

    def _grad_norm(self):
        total_norm = 0
        for p in self.actor_critic.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** (1. / 2)
        return total_norm
    
    def init_meta(self, num_processes):
        skill = np.zeros((num_processes, self.skill_dim), dtype=np.float32)
        skill_id = np.random.choice(self.skill_dim, num_processes)
        for idx, s_id in zip(range(num_processes), skill_id):
            skill[idx, s_id] = 1.0
        skill = torch.Tensor(skill)
        meta = OrderedDict()
        meta['skill'] = skill
        return meta
    
    def update_meta(self):
        skill = np.zeros(self.skill_dim, dtype=np.float32)
        skill[np.random.choice(self.skill_dim)] = 1.0
        skill = torch.Tensor(skill)
        return skill

    def update_discriminator(self, obs, skill):
        metrics = dict()
        in_obs = self.actor_critic.get_encoded_obs(obs)
        loss, df_accuracy = self.compute_discriminator_loss(in_obs, skill)
        self.discriminator_opt.zero_grad()
        loss.backward()
        self.discriminator_opt.step()
        return metrics

    def compute_intrinsic_reward(self, obs, skill):
        in_obs = self.actor_critic.get_encoded_obs(obs)
        z_hat = torch.argmax(skill, dim=1)
        d_pred = self.discriminator(in_obs)
        d_pred_log_softmax = F.log_softmax(d_pred, dim=1)
        _, pred_z = torch.max(d_pred_log_softmax, dim=1, keepdim=True)
        reward = d_pred_log_softmax[torch.arange(d_pred.shape[0]),
                                    z_hat] - math.log(1 / self.skill_dim)
        reward = reward.reshape(-1, 1)
        return reward

    def compute_discriminator_loss(self, in_obs, skill):
        z_hat = torch.argmax(skill, dim=1)
        d_pred = self.discriminator(in_obs)
        d_pred_log_softmax = F.log_softmax(d_pred, dim=1)
        _, pred_z = torch.max(d_pred_log_softmax, dim=1, keepdim=True)
        d_loss = self.discriminator_criterion(d_pred, z_hat)
        df_accuracy = torch.sum(torch.eq(z_hat, pred_z.reshape(1, list(pred_z.size())[0])[0])).float() / list(pred_z.size())[0]
        return d_loss, df_accuracy

    def update(self, rollouts, discard_grad=False):
        
        if rollouts.use_popart:
            value_preds = rollouts.denorm_value_preds
        else:
            value_preds = rollouts.value_preds

        advantages = rollouts.returns[:-1] - value_preds[:-1]
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-5)

        value_loss_epoch = 0
        action_loss_epoch = 0
        dist_entropy_epoch = 0

        if self.log_grad_norm:
            grad_norms = []

        for e in range(self.ppo_epoch):
            
            if self.actor_critic.is_recurrent:
                data_generator = rollouts.recurrent_generator(
                    advantages, self.num_mini_batch
                )
            else:
                data_generator = rollouts.feed_forward_generator(
                    advantages, self.num_mini_batch
                )

            for sample in data_generator:
                obs_batch, recurrent_hidden_states_batch, actions_batch, value_preds_batch, return_batch, masks_batch, old_action_log_probs_batch, adv_targ = sample

                # Update DIAYN discriminator
                # obs_clone = {k: value.clone() for k, value in obs_batch.items()}
                self.update_discriminator(obs_batch, obs_batch['skill'].clone())

                values, action_log_probs, dist_entropy, _ = self.actor_critic.evaluate_actions(
                    obs_batch, recurrent_hidden_states_batch, masks_batch, actions_batch
                )

                ratio = torch.exp(action_log_probs - old_action_log_probs_batch)
                surr1 = ratio * adv_targ
                surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * adv_targ
                action_loss = -torch.min(surr1, surr2).mean()

                if rollouts.use_popart:
                    self.actor_critic.popart.update(return_batch)
                    return_batch = self.actor_critic.popart.normalize(return_batch)

                if self.clip_value_loss:
                    value_pred_clipped = value_preds_batch + (values - value_preds_batch).clamp(-self.clip_param, self.clip_param)
                    value_losses = (values - return_batch).pow(2)
                    value_losses_clipped = (
                        value_pred_clipped - return_batch
                    ).pow(2)
                    value_loss = 0.5 * torch.max(value_losses, value_losses_clipped).mean()
                else:
                    value_loss = F.smooth_l1_loss(values, return_batch)
            
                self.optimizer.zero_grad()
                loss = (value_loss * self.value_loss_coef + action_loss - dist_entropy * self.entropy_coef)
                loss.backward()

                if self.log_grad_norm:
                    grad_norms.append(self._grad_norm())

                if self.max_grad_norm is not None and self.max_grad_norm > 0:
                    nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)

                if not discard_grad:
                    self.optimizer.step()

                value_loss_epoch += value_loss.item()
                action_loss_epoch += action_loss.item()
                dist_entropy_epoch += dist_entropy.item()

        num_updates = self.ppo_epoch * self.num_mini_batch

        value_loss_epoch /= num_updates
        action_loss_epoch /= num_updates
        dist_entropy_epoch /= num_updates

        info = {}
        if self.log_grad_norm:
            info = {'grad_norms': grad_norms}

        return value_loss_epoch, action_loss_epoch, dist_entropy_epoch, info