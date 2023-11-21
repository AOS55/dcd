# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import glob
import os
import shutil
import collections
import timeit
import random

import numpy as np
import torch
from torchvision import utils as vutils

from envs.registration import make as gym_make
from .make_agent import make_agent
from .filewriter import FileWriter
from envs.wrappers import ParallelAdversarialVecEnv, VecMonitor, VecNormalize, \
    VecPreprocessImageWrapper, VecFrameStack, MultiGridFullyObsWrapper, CarRacingWrapper, TimeLimit


class DotDict(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __init__(self, dct):
        for key, value in dct.items():
            if hasattr(value, 'keys'):
                value = DotDict(value)
            self[key] = value

    def __getstate__(self):
        return self

    def __setstate__(self, state):
        self.update(state)
        self.__dict__ = self


def array_to_csv(a):
    return ','.join([str(v) for v in a])


def cprint(condition, *args, **kwargs):
    if condition:
        print(*args, **kwargs)


def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module


def safe_checkpoint(state_dict, path, index=None, archive_interval=None):
    filename, ext = os.path.splitext(path)
    path_tmp = f'{filename}_tmp{ext}'
    torch.save(state_dict, path_tmp)

    os.replace(path_tmp, path)

    if index is not None and archive_interval is not None and archive_interval > 0:
        if index % archive_interval == 0:
            archive_path = f'{filename}_{index}{ext}'
            shutil.copy(path, archive_path)


def cleanup_log_dir(log_dir, pattern='*'):
    try:
        os.makedirs(log_dir)
    except OSError:
        files = glob.glob(os.path.join(log_dir, pattern))
        for f in files:
            os.remove(f)

def seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def save_images(images, path=None, normalize=False, channels_first=False):
    if path is None:
        return

    if isinstance(images, (list, tuple)):
        images = torch.tensor(np.stack(images), dtype=torch.float)
    elif isinstance(images, np.ndarray):
        images = torch.tensor(images, dtype=torch.float)

    if normalize:
        images = images/255

    if not channels_first:
        if len(images.shape) == 4:
            images = images.permute(0,3,1,2)
        else:
            images = images.permute(2,0,1)

    grid = vutils.make_grid(images)
    vutils.save_image(grid, path)


def get_obs_at_index(obs, i):
    if isinstance(obs, dict):
        return {k: obs[k][i] for k in obs.keys()}
    else:
        return obs[i]


def set_obs_at_index(obs, obs_, i):
    if isinstance(obs, dict):
        for k in obs.keys():
            obs[k][i] = obs_[k].squeeze(0)
    else:
        obs[i] = obs_[0].squeeze(0)


def is_discrete_actions(env, adversary=False):
    if adversary:
        return env.adversary_action_space.__class__.__name__ == 'Discrete'
    else:
        return env.action_space.__class__.__name__ == 'Discrete'


def _make_env(cfg):
    env_kwargs = {'seed': cfg.algorithm.seed}
    if cfg.singleton_env:
        env_kwargs.update({
            'fixed_environment': True})
    if cfg.env_name.startswith('CarRacing'):
        env_kwargs.update({
            'n_control_points': cfg.car_racing.num_control_points,
            'min_rad_ratio': cfg.car_racing.min_rad_ratio,
            'max_rad_ratio': cfg.car_racing.max_rad_ratio,
            'use_categorical': cfg.car_racing.use_categorical_adv,
            'use_sketch': cfg.car_racing.use_sketch,
            'clip_reward': cfg.algorithm.clip_reward,
            'sparse_rewards': cfg.algorithm.sparse_rewards,
            'num_goal_bins': cfg.car_racing.num_goal_bins,
        })

    if cfg.env_name.startswith('CarRacing'):
        # Hack: This TimeLimit sandwich allows truncated obs to be passed
        # up the hierarchy with all necessary preprocessing.
        env = gym_make(cfg.env_name, **env_kwargs)
        max_episode_steps = env._max_episode_steps
        reward_shaping = cfg.algorithm.reward_shaping and not cfg.algorithm.sparse_rewards
        assert max_episode_steps % cfg.car_racing.num_action_repeat == 0
        return TimeLimit(CarRacingWrapper(env,
                grayscale=cfg.car_racing.grayscale, 
                reward_shaping=reward_shaping,
                num_action_repeat=cfg.car_racing.num_action_repeat,
                nstack=cfg.car_racing.frame_stack,
                crop=cfg.car_racing.crop_frame), 
            max_episode_steps=max_episode_steps//cfg.car_racing.num_action_repeat)
    elif cfg.env_name.startswith('MultiGrid'):
        env = gym_make(cfg.env_name, **env_kwargs)
        if cfg.use_global_critic or cfg.use_global_policy:
            max_episode_steps = env._max_episode_steps
            env = TimeLimit(MultiGridFullyObsWrapper(env),
                max_episode_steps=max_episode_steps)
        return env
    else:
        return gym_make(cfg.env_name, **env_kwargs)


def create_parallel_env(cfg, adversary=True):
    is_multigrid = cfg.env_name.startswith('MultiGrid')
    is_car_racing = cfg.env_name.startswith('CarRacing')
    is_bipedalwalker = cfg.env_name.startswith('BipedalWalker')

    make_fn = lambda: _make_env(cfg)

    venv = ParallelAdversarialVecEnv([make_fn]*cfg.algorithm.num_processes, adversary=adversary)
    venv = VecMonitor(venv=venv, filename=None, keep_buf=100)
    venv = VecNormalize(venv=venv, ob=False, ret=cfg.algorithm.normalize_returns)

    obs_key = None
    scale = None
    transpose_order = [2,0,1] # Channels first
    if is_multigrid:
        obs_key = 'image'
        scale = 10.0

    if is_car_racing:
        ued_venv = VecPreprocessImageWrapper(venv=venv) # move to tensor

    if is_bipedalwalker:
        transpose_order = None

    venv = VecPreprocessImageWrapper(venv=venv, obs_key=obs_key,
            transpose_order=transpose_order, scale=scale)

    if is_multigrid or is_bipedalwalker:
        ued_venv = venv

    if cfg.singleton_env:
        seeds = [cfg.algorithm.seed]*cfg.algorithm.num_processes
    else:
        seeds = [i for i in range(cfg.algorithm.num_processes)]
    venv.set_seed(seeds)

    return venv, ued_venv


def is_dense_reward_env(env_name):
    if env_name.startswith('CarRacing'):
        return True
    else:
        return False


def make_plr_args(cfg, obs_space, action_space):
    return dict( 
        seeds=[], 
        obs_space=obs_space, 
        action_space=action_space, 
        num_actors=cfg.algorithm.num_processes,
        strategy=cfg.plr.level_replay_strategy,
        replay_schedule=cfg.plr.level_replay_schedule,
        score_transform=cfg.plr.level_replay_score_transform,
        temperature=cfg.plr.level_replay_temperature,
        eps=cfg.plr.level_replay_eps,
        rho=cfg.plr.level_replay_rho,
        replay_prob=cfg.plr.level_replay_prob, 
        alpha=cfg.plr.level_replay_alpha,
        staleness_coef=cfg.plr.staleness_coef,
        staleness_transform=cfg.plr.staleness_transform,
        staleness_temperature=cfg.plr.staleness_temperature,
        sample_full_distribution=cfg.plr.train_full_distribution,
        seed_buffer_size=cfg.plr.level_replay_seed_buffer_size,
        seed_buffer_priority=cfg.plr.level_replay_seed_buffer_priority,
        use_dense_rewards=is_dense_reward_env(cfg.env_name),
        gamma=cfg.algorithm.gamma
    )
