import sys
from typing import OrderedDict
import hydra
import os
import csv
import json
import pickle
from collections import defaultdict
from tqdm import tqdm
from itertools import cycle
import glob

import time

import torch
import numpy as np
import moviepy.editor as mpy
import matplotlib as mpl
import matplotlib.pyplot as plt
from baselines.logger import HumanOutputFormat


from envs.registration import make as gym_make
from envs.multigrid import *
from envs.multigrid.adversarial import *
from envs.box2d import *
from envs.bipedalwalker import *
from envs.runners.adversarial_runner import AdversarialRunner
from util import make_agent, FileWriter, safe_checkpoint, create_parallel_env, make_plr_args, save_images, DotDict, is_discrete_actions, str2bool
from eval import Evaluator
from evaluation import WriteEvaluator
from envs.wrappers import VecMonitor, VecPreprocessImageWrapper, ParallelAdversarialVecEnv, \
	MultiGridFullyObsWrapper, VecFrameStack, CarRacingWrapper


class Workspace:

    def __init__(self, cfg):

        self.display = None

        if sys.platform.startswith('linux'):
            print(f'setting up virtual display NOW!')
            
            import pyvirtualdisplay
            self.display = pyvirtualdisplay.Display(visible=0, size=(1400, 900), color_depth=24)
            self.display.start()

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

        # Setup results management
        self.result_path = os.path.join(*["..", "..", "..", self.cfg.result_path])
        os.makedirs(self.result_path, exist_ok=True)
        if self.cfg.prefix is not None:
            self.result_fname = self.cfg.prefix
        else:
            self.result_fname = self.cfg.xpid
        self.result_fname = f"{self.result_fname}-group_eval-{self.cfg.model_name}"
        self.result_fpath = os.path.join(self.result_path, self.result_fname)
        
        self.result_fpath = f'{self.result_fpath}.csv'
        csvout = open(self.result_fpath, 'w', newline='')
        csvwriter = csv.writer(csvout)

        env_results = defaultdict(list)
        env_names = self.cfg.test_env_names
        num_seeds = 0
        checkpoint_nums = []

        xpid = self.cfg.xpid
        xpid_dir = os.path.join(base_path, xpid)
        models = glob.glob(os.path.join(xpid_dir, "*.tar"))
        meta_json_path = os.path.join(xpid_dir, 'meta.json')

        meta_json_file = open(meta_json_path)
        meta_cfg = json.load(meta_json_file)
        meta_dict = DotDict(meta_cfg['args'])
        make_fn = [lambda: WriteEvaluator.make_env(env_names[0])]
        dummy_venv = ParallelAdversarialVecEnv(make_fn, adversary=False, is_eval=True)
        dummy_venv = WriteEvaluator.wrap_venv(dummy_venv, env_name=env_names[0], device=self.device)
        agent = make_agent(name='agent', env=dummy_venv, cfg=meta_dict, device=self.device)

        env_names_ = env_names

        idm = 0
        for model in models:
            if os.path.exists(model):
                
                a_time = time.time()

                print(f'model: {model}')
                print(f'idm: {idm}')
                idm += 1                
                checkpoint_num = model.split("/")[-1].split(".")[0].split("_")[-1]
                if checkpoint_num == "model":
                    checkpoint_num = '0'
                print(f'checkpoint_num: {checkpoint_num}')

                checkpoint_nums.append(checkpoint_num)
                
                # Load the agent
                try:
                    checkpoint = torch.load(model, map_location='cpu')
                except:
                    print(f'checkpoint: {model} not found')
                    continue
                model_name = self.cfg.model_name

                b_time = time.time()

                if 'runner_state_dict' in checkpoint:
                    agent.algo.actor_critic.load_state_dict(checkpoint['runner_state_dict']['agent_state_dict'][model_name])
                else:
                    agent.algo.actor_critic.load_state_dict(checkpoint)

                c_time = time.time()

                # env_names_ = env_names[start_idx:start_idx+chunk_size]
                evaluator = WriteEvaluator(env_names_, 
                    num_processes=self.cfg.num_processes, 
                    num_episodes=self.cfg.num_episodes, 
                    frame_stack=self.cfg.frame_stack,
                    grayscale=self.cfg.grayscale,
                    use_global_critic=self.cfg.use_global_critic,
                    record_video=self.cfg.record_video)

                # Evaluate the model
                # xpid_flags.update(args)
                # xpid_flags.update({"use_skip": False})
                
                d_time = time.time()

                stats, pos_dict, frames_dict = evaluator.evaluate(agent,
                    plot_pos=self.cfg.plot_pos, 
                    deterministic=self.cfg.deterministic, 
                    show_progress=self.cfg.verbose,
                    render=self.cfg.render,
                    accumulator=self.cfg.accumulator,
                    reward_free=self.cfg.reward_free)

                e_time = time.time()

                for k, v in stats.items():
                    if self.cfg.accumulator:
                        env_results[k].append(v)
                    else:
                        env_results[k] += v
                
                f_time = time.time()
                tot_time = f_time - a_time

                print(f'TIMES: load agent: {b_time-a_time}, load state dict: {c_time - b_time}, WriteEvaluator: {d_time - c_time}, evaluate: {e_time - d_time}, append: {f_time-e_time}')
                print(f'FractionTime: load agent: {(b_time-a_time)/tot_time}, load state dict: {(c_time - b_time)/tot_time}, WriteEvaluator: {(d_time - c_time)/tot_time}, evaluate: {(e_time - d_time)/tot_time}, append: {(f_time-e_time)/tot_time}')


                # Store plots
                evaluator.close()
            else:
                print(f'No model path {model}')

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

            # if self.cfg.accumulator:
            #     csvwriter.writerow(['metric',] + [x for x in range(num_seeds)])
            # else:
            #     csvwriter.writerow(['metric',] + [x for x in range(num_seeds*self.cfg.num_episodes)])
        csvwriter.writerow(['metric'] + checkpoint_nums)
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

        if self.display:
            self.display.stop()

@hydra.main(config_path='conf/eval/.', config_name='bipedal_dr', version_base="1.1")
def main(cfg):
    from group_evaluation import Workspace as W
    workspace = W(cfg)
    
if __name__ == "__main__":
    main()
