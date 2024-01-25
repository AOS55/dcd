import sys
from typing import OrderedDict
import hydra
import os
import csv
import json
import fnmatch
import re
import time
import timeit
import logging
import pickle
from collections import defaultdict
from tqdm import tqdm
from itertools import cycle

import torch
import numpy as np
import wandb
import moviepy.editor as mpy
from omegaconf import OmegaConf
import gym
import matplotlib as mpl
import matplotlib.pyplot as plt
from baselines.logger import HumanOutputFormat

display = None

if sys.platform.startswith('linux'):
    print('Setting up virtual display')

    import pyvirtualdisplay
    display = pyvirtualdisplay.Display(visible=0, size=(1400, 900), color_depth=24)
    display.start()

from envs.registration import make as gym_make
from envs.multigrid import *
from envs.multigrid.adversarial import *
from envs.box2d import *
from envs.bipedalwalker import *
from envs.runners.adversarial_runner import AdversarialRunner 
from util import make_agent, FileWriter, safe_checkpoint, create_parallel_env, make_plr_args, save_images, DotDict, is_discrete_actions, str2bool
from eval import Evaluator
from envs.wrappers import VecMonitor, VecPreprocessImageWrapper, ParallelAdversarialVecEnv, \
	MultiGridFullyObsWrapper, VecFrameStack, CarRacingWrapper


class WriteEvaluator:
    def __init__(self, 
        env_names, 
        num_processes, 
        num_episodes=10, 
        record_video=False, 
        device='cpu',
        **kwargs):
        self.kwargs = kwargs # kwargs for env wrappers
        self._init_parallel_envs(
            env_names, num_processes, device=device, record_video=record_video, **kwargs)
        self.num_episodes = num_episodes
        if 'Bipedal' in env_names[0]:
            self.solved_threshold = 230
        else:
            self.solved_threshold = 0

    def get_stats_keys(self):
        keys = []
        for env_name in self.env_names:
            keys += [f'solved_rate:{env_name}', f'test_returns:{env_name}']
        return keys

    @staticmethod
    def make_env(env_name, record_video=False, **kwargs):
        if env_name in ['BipedalWalker-v3', 'BipedalWalkerHardcore-v3']:
            env = gym.make(env_name)
        else:
            env = gym_make(env_name)

        is_multigrid = env_name.startswith('MultiGrid')
        is_car_racing = env_name.startswith('CarRacing')

        if is_car_racing:
            grayscale = kwargs.get('grayscale', False)
            num_action_repeat = kwargs.get('num_action_repeat', 8)
            nstack = kwargs.get('frame_stack', 4)
            crop = kwargs.get('crop_frame', False)

            env = CarRacingWrapper(
                env=env,
                grayscale=grayscale, 
                reward_shaping=False,
                num_action_repeat=num_action_repeat,
                nstack=nstack,
                crop=crop,
                eval_=True)

            if record_video:
                from gym.wrappers.monitor import Monitor
                env = Monitor(env, "videos/", force=True)
                print('Recording video!', flush=True)

        if is_multigrid and kwargs.get('use_global_policy'):
            env = MultiGridFullyObsWrapper(env, is_adversarial=False)

        return env

    @staticmethod
    def wrap_venv(venv, env_name, device='cpu'):
        is_multigrid = env_name.startswith('MultiGrid') or env_name.startswith('MiniGrid')
        is_car_racing = env_name.startswith('CarRacing')
        is_bipedal = env_name.startswith('BipedalWalker')

        obs_key = None
        scale = None
        if is_multigrid:
            obs_key = 'image'
            scale = 10.0

        # Channels first
        transpose_order = [2,0,1]

        if is_bipedal:
            transpose_order = None

        venv = VecMonitor(venv=venv, filename=None, keep_buf=100)
        venv = VecPreprocessImageWrapper(venv=venv, obs_key=obs_key,
                transpose_order=transpose_order, scale=scale, device=device)

        return venv

    def _init_parallel_envs(self, env_names, num_processes, device=None, record_video=False, **kwargs):
        self.env_names = env_names
        self.num_processes = num_processes
        self.device = device
        self.venv = {env_name:None for env_name in env_names}

        make_fn = []
        for env_name in env_names:
            make_fn = [lambda: WriteEvaluator.make_env(env_name, record_video, **kwargs)]*self.num_processes
            venv = ParallelAdversarialVecEnv(make_fn, adversary=False, is_eval=True)
            venv = WriteEvaluator.wrap_venv(venv, env_name, device=device)
            self.venv[env_name] = venv

        self.is_discrete_actions = is_discrete_actions(self.venv[env_names[0]])

    def close(self):
        for _, venv in self.venv.items():
            venv.close()

    def evaluate(self, 
        agent, 
        reward_free=False,
        deterministic=False, 
        show_progress=False,
        render=False,
        plot_pos=False,
        accumulator='mean'):

        # Evaluate agent for N episodes
        venv = self.venv
        env_returns = {}
        env_solved_episodes = {}
        
        if render:
            frames_dict = {}
        else:
            frames_dict = None

        if plot_pos:
            pos_dict = {}
        else:
            pos_dict = None

        for env_name, venv in self.venv.items():
            
            # Preallocate lists
            returns = []
            solved_episodes = 0
            idp = 0
            idf = 0
            
            # Verify environment type
            is_multigrid = env_name.startswith('MultiGrid') or env_name.startswith('MiniGrid')
            is_car_racing = env_name.startswith('CarRacing')
            is_bipedal = env_name.startswith('BipedalWalker')

            obs = venv.reset()

            frames = []
            positions = []

            recurrent_hidden_states = torch.zeros(
                self.num_processes, agent.algo.actor_critic.recurrent_hidden_state_size, device=self.device)
            if agent.algo.actor_critic.is_recurrent and agent.algo.actor_critic.rnn.arch == 'lstm':
                recurrent_hidden_states = (recurrent_hidden_states, torch.zeros_like(recurrent_hidden_states))
            masks = torch.ones(self.num_processes, 1, device=self.device)

            # Init Meta in obs        
            if reward_free:
                skill_cycle = cycle(range(agent.algo.skill_dim))
                skill = np.zeros((self.num_processes, agent.algo.skill_dim), dtype=np.float32)
                ids = next(skill_cycle)
                for idx in zip(range(self.num_processes)):
                    skill[idx, ids] = 1.0
                skill = torch.Tensor(skill)
                meta = OrderedDict()
                meta['skill'] = skill
                skill = meta
                # skill = agent.algo.init_meta(self.num_processes)
                if type(obs) == dict:
                    obs = {**obs, **skill}
                else:
                    obs = torch.concat((obs, skill['skill'].to(self.device)), axis=1)

            pbar = None
            if show_progress:
                pbar = tqdm(total=self.num_episodes)

            while len(returns) < self.num_episodes:
                # Sample actions
                with torch.no_grad():
                    _, action, _, recurrent_hidden_states = agent.act(
                        obs, recurrent_hidden_states, masks, deterministic=deterministic)

                # _, _, a_log, _ = agent.act(obs, recurrent_hidden_states, masks, deterministic=False)
                # print(f"action_log: {a_log}")

                # Observe reward and next obs
                action = action.cpu().numpy()
                if not self.is_discrete_actions:
                    action = agent.process_action(action)
                obs, reward, done, infos = venv.step(action)

                if reward_free:
                    if type(obs) == dict:
                        obs = {**obs, **skill}
                    else:
                        obs = torch.concat((obs, skill['skill'].to(self.device)), axis=1)

                masks = torch.tensor(
                    [[0.0] if done_ else [1.0] for done_ in done],
                    dtype=torch.float32,
                    device=self.device)

                for i, info in enumerate(infos):
                    if 'episode' in info.keys():
                        returns.append(info['episode']['r'])
                        if returns[-1] > self.solved_threshold:
                            solved_episodes += 1
                        if pbar:
                            pbar.update(1)

                        # zero hidden states
                        if agent.is_recurrent:
                            recurrent_hidden_states[0][i].zero_()
                            recurrent_hidden_states[1][i].zero_()

                        if len(returns) >= self.num_episodes:
                            break
                
                if reward_free:
                    for idx in range(masks.shape[1]):
                        if masks[idx] == 0.0:
                            ids = next(skill_cycle)
                            next_skill = np.zeros(agent.algo.skill_dim, dtype=np.float32)
                            next_skill[ids] = 1.0
                            next_skill = torch.Tensor(next_skill)
                            skill["skill"][idx] = next_skill
                            # skill["skill"][idx] = agent.algo.update_meta()

                if render:
                    frame_array = np.array(venv.get_rgb_images()[0])
                    frames.append(frame_array)
                    if masks[0] == 0.0:
                        # frames_list.append(frames)
                        idf += 1
                        frames = np.array(frames)
                        if reward_free:
                            if type(obs) == dict:
                                frames_dict[f"{env_name}_i{idf}_s{torch.argmax(obs['skill'][0]).item()}"] = frames
                            else:
                                frames_dict[f"{env_name}_i{idf}_s{torch.argmax(obs[0, -agent.algo.skill_dim:])}"] = frames
                        else:
                            frames_dict[f"{env_name}_i{idf}"] = frames
                        frames = []

                if is_multigrid and plot_pos:
                    pos = venv.get_agent_pos()[0][0][0]
                    positions.append(pos)
                    if masks[0] == 0.0:
                        idp += 1
                        positions = np.array(positions)
                        if reward_free:
                            if type(obs) == dict:
                                pos_dict[f"{env_name}_i{idp}_s{torch.argmax(obs['skill'][0]).item()}"] = positions
                            else:
                                pos_dict[f"{env_name}_i{idp}_s{torch.argmax(obs[0, -agent.algo.skill_dim:])}"] = positions
                        else:
                            pos_dict[f"{env_name}_i{idp}"] = positions
                        positions = []

            if pbar:
                pbar.close()	

            env_returns[env_name] = returns
            env_solved_episodes[env_name] = solved_episodes

        stats = {}
        for env_name in self.env_names:
            if accumulator == 'mean':
                stats[f"solved_rate:{env_name}"] = env_solved_episodes[env_name]/self.num_episodes

            if accumulator == 'mean':
                stats[f"test_returns:{env_name}"] = np.mean(env_returns[env_name])
            else:
                stats[f"test_returns:{env_name}"] = env_returns[env_name]

        # if render:
        #     for name, frames in frames_dict.items():
        #         frames = np.array(frames)
        #         print(f'frames.shape: {frames.shape}')
        #         frames = np.moveaxis(frames, [0, 1, 2, 3], [0, -2, -1, -3])
        #         print(f'frames.shape: {frames.shape}')
        #         # Write to the storage dir
        #         # wandb.log({f"video/{name}": wandb.Video(frames, fps=16)})

        return stats, pos_dict, frames_dict


class Workspace:

    def __init__(self, cfg):
        
        os.environ["OMP_NUM_THREADS"] = "1"
        self.cfg = cfg

        self.cfg.num_processes = min(self.cfg.num_processes, self.cfg.num_episodes)

        # === Determine device ===
        self.cuda = not self.cfg.no_cuda and torch.cuda.is_available()
        self.device = torch.device("cuda:0" if self.cuda else "cpu")
        if 'cuda' in self.device.type:
            torch.backends.cudnn.benchmark = True
            print('Using CUDA\n')

        # === Load checkpoint ===
        base_path = os.path.expandvars(os.path.expanduser(self.cfg.log_dir))

        # Set up results management
        self.result_path = os.path.join(*["..", "..", "..", self.cfg.result_path])
        os.makedirs(self.result_path, exist_ok=True)
        if self.cfg.prefix is not None:
            self.result_fname = self.cfg.prefix
        else:
            self.result_fname = self.cfg.xpid
        self.result_fname = f"{self.result_fname}-{self.cfg.model_tar}-{self.cfg.model_name}"
        self.result_fpath = os.path.join(self.result_path, self.result_fname)
        # if os.path.exists(f'{result_fpath}.csv'):
        #     result_fpath = os.path.join(self.cfg.result_path, f'{result_fname}_redo')
        self.result_fpath = f'{self.result_fpath}.csv'

        csvout = open(self.result_fpath, 'w', newline='')
        csvwriter = csv.writer(csvout)

        env_results = defaultdict(list)

        env_names = self.cfg.test_env_names

        num_envs = len(env_names)
        if num_envs*self.cfg.num_processes > self.cfg.max_num_processes:
            chunk_size = self.cfg.max_num_processes//self.cfg.num_processes
        else:
            chunk_size = num_envs
            
        num_chunks = int(np.ceil(num_envs/chunk_size))

        if self.cfg.record_video:
            num_chunks = 1
            chunk_size = 1
            self.cfg.num_processes = 1

        num_seeds = 0
        xpid = self.cfg.xpid

        xpid_dir = os.path.join(base_path, xpid)
        meta_json_path = os.path.join(xpid_dir, 'meta.json')

        model_tar = f'{self.cfg.model_tar}.tar'
        checkpoint_path = os.path.join(xpid_dir, model_tar)

        if os.path.exists(checkpoint_path):
            meta_json_file = open(meta_json_path)
            meta_cfg = json.load(meta_json_file)
            meta_dict = DotDict(meta_cfg['args'])
            # xpid_flags = DotDict(json.load(meta_json_file)['args'])
            
            make_fn = [lambda: WriteEvaluator.make_env(env_names[0])]
            dummy_venv = ParallelAdversarialVecEnv(make_fn, adversary=False, is_eval=True)
            dummy_venv = WriteEvaluator.wrap_venv(dummy_venv, env_name=env_names[0], device=self.device)

            # Load the agent
            agent = make_agent(name='agent', env=dummy_venv, cfg=meta_dict, device=self.device)

            try:
                checkpoint = torch.load(checkpoint_path, map_location='cpu')
            except:
                print(f'checkpoint: {checkpoint_path} not found')
                # continue
            model_name = self.cfg.model_name

            if 'runner_state_dict' in checkpoint:
                agent.algo.actor_critic.load_state_dict(checkpoint['runner_state_dict']['agent_state_dict'][model_name])
            else:
                agent.algo.actor_critic.load_state_dict(checkpoint)

            num_seeds += 1

			# Evaluate environment batch in increments of chunk size
            for i in range(num_chunks):
                start_idx = i*chunk_size
                # env_names_ = env_names[start_idx:start_idx+chunk_size]
                env_names_ = env_names

                # Evaluate the model
                # xpid_flags.update(args)
                # xpid_flags.update({"use_skip": False})

                evaluator = WriteEvaluator(env_names_, 
                    num_processes=self.cfg.num_processes, 
                    num_episodes=self.cfg.num_episodes, 
                    frame_stack=self.cfg.frame_stack,
                    grayscale=self.cfg.grayscale,
                    use_global_critic=self.cfg.use_global_critic,
                    record_video=self.cfg.record_video)

                stats, pos_dict, frames_dict = evaluator.evaluate(agent,
                    plot_pos=self.cfg.plot_pos, 
                    deterministic=self.cfg.deterministic, 
                    show_progress=self.cfg.verbose,
                    render=self.cfg.render,
                    accumulator=self.cfg.accumulator,
                    reward_free=self.cfg.reward_free)

                for k,v in stats.items():
                    if self.cfg.accumulator:
                        env_results[k].append(v)
                    else:
                        env_results[k] += v
                
                # Store plots

                evaluator.close()
        else:
            print(f'No model path {checkpoint_path}')

        output_results = {}
        
        for k,_ in stats.items():
            results = env_results[k]
            output_results[k] = f'{np.mean(results):.2f} +/- {np.std(results):.2f}'
            q1 = np.percentile(results, 25, interpolation='midpoint')
            q3 = np.percentile(results, 75, interpolation='midpoint')
            median = np.median(results)
            output_results[f'iq_{k}'] = f'{q1:.2f}--{median:.2f}--{q3:.2f}'
            print(f"{k}: {output_results[k]}")
        HumanOutputFormat(sys.stdout).writekvs(output_results)

        if self.cfg.accumulator:
            csvwriter.writerow(['metric',] + [x for x in range(num_seeds)])
        else:
            csvwriter.writerow(['metric',] + [x for x in range(num_seeds*self.cfg.num_episodes)])
        for k,v in env_results.items():
            row = [k,] + v
            csvwriter.writerow(row)

        # Write out pos dict
        if self.cfg.plot_pos:
            pos_path = os.path.join(self.result_path, f"{self.result_fname}-positions.pkl")
            with open(pos_path, 'wb') as fp:
                pickle.dump(pos_dict, fp)

        # Write out frames dict
        if self.cfg.record_video:
            frame_path = os.path.join(self.result_path, f"{self.result_fname}-frames.pkl")
            with open(frame_path, 'wb') as fp:
                pickle.dump(frames_dict, fp) 

        if display:
            display.stop()

    def make_pos_graph(self):
        
        colours = [[0, 18, 25], [0, 95, 115], [10, 147, 150], [148, 210, 189], [233, 216, 166],
               [238, 155, 0], [202, 103, 2], [187, 62, 3], [174, 32, 18], [155, 34, 38]]
        colours = [[value/255 for value in rgb] for rgb in colours]

        fig, ax = plt.subplots()
        pos_path = os.path.join(self.result_path, f"{self.result_fname}-positions.pkl")
        with open(pos_path, 'rb') as f:
            positions = pickle.load(f)
        
        for key, value in positions.items():
            x = value[:-1, 0]
            y = value[:-1, 1]
            ax.plot(x, y, label=key)
            # [print(f"{x_val}, {y_val}") for x_val, y_val in zip(x, y)]
        
        plt.savefig(os.path.join(self.result_path, f"{self.result_fname}-pos_plot.pdf"))
    
    def make_videos(self, fps=30):

        frames_path = os.path.join(self.result_path, f"{self.result_fname}-frames.pkl")
        with open(frames_path, 'rb') as f:
            frames = pickle.load(f)
        
        for key, value in frames.items():
            clip = mpy.ImageSequenceClip(list(value), fps=fps)
            clip.write_gif(os.path.join(self.result_path, f"{self.result_fname}-{key}-movie.gif"))


@hydra.main(config_path='conf/eval/.', config_name='bipedal_diayn_dr', version_base="1.1")
def main(cfg):
    from evaluation import Workspace as W
    workspace = W(cfg)
    # workspace.make_pos_graph()
    workspace.make_videos()

if __name__ == "__main__":
    main()
