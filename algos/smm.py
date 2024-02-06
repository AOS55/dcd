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


class VAE(nn.Module):
    def __init__(self, obs_dim, z_dim, code_dim, vae_beta, device):
        super().__init__()
        self.z_dim = z_dim
        self.code_dim = code_dim
        
        self.make_networks(obs_dim, z_dim, code_dim)
        self.beta = vae_beta

        self.apply(weight_init)
        self.device = device

    def make_networks(self, obs_dim, z_dim, code_dim):
        self.enc = nn.Sequential(nn.Linear(obs_dim + z_dim, 150), nn.ReLU(),
                                 nn.Linear(150, 150), nn.ReLU())
        self.enc_mu = nn.Linear(150, code_dim)
        self.enc_logvar = nn.Linear(150, code_dim)
        self.dec = nn.Sequential(nn.Linear(code_dim, 150), nn.ReLU(),
                                 nn.Linear(150, 150), nn.ReLU(),
                                 nn.Linear(150, obs_dim + z_dim))

    def encode(self, obs_z):
        enc_features = self.enc(obs_z)
        mu = self.enc_mu(enc_features)
        logvar = self.enc_logvar(enc_features)
        stds = (0.5 * logvar).exp()
        return mu, logvar, stds

    def forward(self, obs_z, epsilon):
        mu, logvar, stds = self.encode(obs_z)
        code = epsilon * stds + mu
        obs_distr_params = self.dec(code)
        return obs_distr_params, (mu, logvar, stds)

    def loss(self, obs_z):
        epsilon = torch.randn([obs_z.shape[0], self.code_dim]).to(self.device)
        obs_distr_params, (mu, logvar, stds) = self(obs_z, epsilon)
        kle = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(),
                               dim=1).mean()
        log_prob = F.mse_loss(obs_z, obs_distr_params, reduction='none')

        loss = self.beta * kle + log_prob.mean()
        return loss, log_prob.sum(list(range(1, len(log_prob.shape)))).view(
            log_prob.shape[0], 1)



class SMMDiscriminator(nn.Module):

    def __init__(self, obs_dim, z_dim, hidden_dim, vae_beta, device):
        super().__init__()
        self.z_dim = z_dim
        self.z_pred_net = nn.Sequential(nn.Linear(obs_dim, hidden_dim),
                                        nn.ReLU(),
                                        nn.Linear(hidden_dim, hidden_dim),
                                        nn.ReLU(),
                                        nn.Linear(hidden_dim, z_dim))
        self.vae = VAE(
            obs_dim=obs_dim,
            z_dim=z_dim,
            code_dim=128,
            vae_beta=vae_beta,
            device=device
        )
        self.apply(weight_init)

        print(f"Density Model --> {self.vae}")
        print(f"Discriminator Model --> {self.z_pred_net}")

    def predict_logits(self, obs):
        z_pred_logits = self.z_pred_net(obs)
        return z_pred_logits

    def loss(self, logits, z):
        z_labels = torch.argmax(z, 1)
        return nn.CrossEntropyLoss(reduction='none')(logits, z_labels)
    

class SMM:

    def __init__(self,
                 actor_critic,
                 discriminator,
                 skill_dim,
                 clip_param,
                 ppo_epoch,
                 num_mini_batch,
                 value_loss_coef,
                 entropy_coef,
                 discriminator_dim,
                 lr=None,
                 eps=None,
                 max_grad_norm=None,
                 clip_value_loss=True,
                 log_grad_norm=False):
        
        # Actor critic model
        self.actor_critic = actor_critic

        # Discriminator model
        self.discriminator = discriminator
        self.discriminator_dim = discriminator_dim
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

        # Setup optimizer for PPO
        self.optimizer = optim.Adam(actor_critic.parameters(), lr=lr, eps=eps)

        # SMM/Discriminator optimizers
        self.pred_optimizer = torch.optim.Adam(self.discriminator.z_pred_net.parameters(), lr=lr)
        self.vae_optimizer = torch.optim.Adam(self.discriminator.vae.parameters(), lr=lr)
        
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
        obs_z = torch.cat([in_obs, skill], dim=1) 
        vae_metric, h_s_z = self.update_vae(obs_z)
        pred_metrics, h_z_s = self.update_pred(in_obs, skill)
        metrics.update(vae_metric) 
        metrics.update(pred_metrics)
        return h_s_z, h_z_s, metrics

    def update_vae(self, obs_z):
        metrics = dict()
        loss, h_s_z = self.discriminator.vae.loss(obs_z)
        self.vae_optimizer.zero_grad()
        loss.backward()
        self.vae_optimizer.step()
        metrics['loss_vae'] = loss.cpu().item()
        return metrics, h_s_z
    
    def update_pred(self, obs, z):
        metrics = dict()
        logits = self.discriminator.predict_logits(obs)
        h_z_s = self.discriminator.loss(logits, z).unsqueeze(-1)
        loss = h_z_s.mean()
        self.pred_optimizer.zero_grad()
        loss.backward()
        self.pred_optimizer.step()
        metrics['loss_pred'] = loss.cpu().item()
        return metrics, h_z_s
    
    def compute_intrinsic_reward(self, obs, skill, extr_reward):
        metrics = dict()
        in_obs = self.actor_critic.get_encoded_obs(obs)
        in_skill = skill.to(self.actor_critic.device)
        obs_z = torch.cat([in_obs, in_skill], dim=1)
        h_z = np.log(self.skill_dim)  # One-hot z encoding
        h_z *= torch.ones_like(extr_reward).to(self.actor_critic.device)
        _, h_s_z = self.discriminator.vae.loss(obs_z)
        logits = self.discriminator.predict_logits(in_obs)
        h_z_s = self.discriminator.loss(logits, in_skill).unsqueeze(-1)
        
        reward = 1000.0 * extr_reward.to(self.actor_critic.device) + h_s_z.detach() + h_z + h_z_s.detach() # Currently has no scaling
        metrics['extr_reward'] = extr_reward.detach().to(device='cpu')
        metrics['h_s_z'] = h_s_z.detach().to(device='cpu')
        metrics['h_z'] = h_z.detach().to(device='cpu')
        metrics['h_z_s'] = h_z_s.detach().to(device='cpu')

        return reward, metrics

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

                if type(obs_batch) == dict:
                    h_s_z, h_z_s, metrics = self.update_discriminator(obs_batch, obs_batch['skill'].clone())
                else:
                    self.update_discriminator(obs_batch[:, :self.discriminator_dim], obs_batch[:, -self.skill_dim:])
                
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