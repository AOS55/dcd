# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import sys
import hydra
import os
import time
import timeit
import logging
from arguments import parser

import torch
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

from envs.multigrid import *
from envs.multigrid.adversarial import *
from envs.box2d import *
from envs.bipedalwalker import *
from envs.runners.adversarial_runner import AdversarialRunner 
from util import make_agent, FileWriter, safe_checkpoint, create_parallel_env, make_plr_args, save_images
from eval import Evaluator

class Workspace:

    def __init__(self, cfg):
        os.environ["OMP_NUM_THREADS"] = "1"

        self.cfg = cfg
        # args = parser.parse_args()
        
        # === Configure logging ==
        if self.cfg.logging.xpid is None:
            self.cfg.logging.xpid = "lr-%s" % time.strftime("%Y%m%d-%H%M%S")
        self.log_dir = os.path.expandvars(os.path.expanduser(self.cfg.logging.log_dir))
        self.filewriter = FileWriter(
            xpid=self.cfg.logging.xpid, xp_args=self.cfg.__dict__, rootdir=self.log_dir
        )
        self.screenshot_dir = os.path.join(self.log_dir, self.cfg.logging.xpid, 'screenshots')
        if not os.path.exists(self.screenshot_dir):
            os.makedirs(self.screenshot_dir, exist_ok=True)
        if self.cfg.logging.verbose:
            logging.getLogger().setLevel(logging.INFO)
        else:
            logging.disable(logging.CRITICAL)

        # === Determine device ====
        self.cuda = not self.cfg.no_cuda and torch.cuda.is_available()
        self.device = torch.device("cuda:0" if self.cuda else "cpu")
        if 'cuda' in self.device.type:
            torch.backends.cudnn.benchmark = True
            print('Using CUDA\n')

        # === Create parallel envs ===
        self.venv, self.ued_venv = create_parallel_env(self.cfg)

        is_training_env = self.cfg.ued.ued_algo in ['paired', 'flexible_paired', 'minimax']
        is_paired = self.cfg.ued.ued_algo in ['paired', 'flexible_paired']

        self.agent = make_agent(name='agent', env=self.venv, cfg=self.cfg, device=self.device)
        self.adversary_agent, self.adversary_env = None, None
        if is_paired:
            self.adversary_agent = make_agent(name='adversary_agent', env=self.venv, cfg=self.cfg, device=self.device)

        if is_training_env:
            self.adversary_env = make_agent(name='adversary_env', env=self.venv, cfg=self.cfg, device=self.device)
        if self.cfg.ued.ued_algo == 'domain_randomization' and self.cfg.plr.use_plr and not self.cfg.ued.use_reset_random_dr:
            self.adversary_env = make_agent(name='adversary_env', env=self.venv, cfg=self.cfg, device=self.device)
            self.adversary_env.random()

        # === Create runner ===
        plr_args = None
        if self.cfg.plr.use_plr:
            plr_args = make_plr_args(self.cfg, self.venv.observation_space, self.venv.action_space)
        self.train_runner = AdversarialRunner(
            cfg=self.cfg,
            venv=self.venv,
            agent=self.agent, 
            ued_venv=self.ued_venv, 
            adversary_agent=self.adversary_agent,
            adversary_env=self.adversary_env,
            flexible_protagonist=False,
            train=True,
            plr_args=plr_args,
            device=self.device)

        # === Configure checkpointing ===
        self.timer = timeit.default_timer
        self.initial_update_count = 0
        self.last_logged_update_at_restart = -1
        self.checkpoint_path = os.path.expandvars(
            os.path.expanduser("%s/%s/%s" % (self.log_dir, self.cfg.logging.xpid, "model.tar"))
        )
        ## This is only used for the first iteration of finetuning
        if self.cfg.xpid_finetune:
            model_fname = f'{self.cfg.model_finetune}.tar'
            self.base_checkpoint_path = os.path.expandvars(
                os.path.expanduser("%s/%s/%s" % (self.log_dir, self.cfg.xpid_finetune, model_fname))
            )

        # === Load checkpoint ===
        if self.cfg.logging.checkpoint and os.path.exists(self.checkpoint_path):
            self.checkpoint_states = torch.load(self.checkpoint_path, map_location=lambda storage, loc: storage)
            self.last_logged_update_at_restart = self.filewriter.latest_tick() # ticks are 0-indexed updates
            self.train_runner.load_state_dict(self.checkpoint_states['runner_state_dict'])
            self.initial_update_count = self.train_runner.num_updates
            logging.info(f"Resuming preempted job after {self.initial_update_count} updates\n") # 0-indexed next update
        elif self.cfg.xpid_finetune and not os.path.exists(self.checkpoint_path):
            self.checkpoint_states = torch.load(self.base_checkpoint_path)
            self.state_dict = self.checkpoint_states['runner_state_dict']
            self.agent_state_dict = self.state_dict.get('agent_state_dict')
            self.optimizer_state_dict = self.state_dict.get('optimizer_state_dict')
            self.train_runner.agents['agent'].algo.actor_critic.load_state_dict(self.agent_state_dict['agent'])
            self.train_runner.agents['agent'].algo.optimizer.load_state_dict(self.optimizer_state_dict['agent'])

        # === Set up Evaluator ===
        self.evaluator = None
        if self.cfg.test_env_names:
            self.evaluator = Evaluator(
                self.cfg.test_env_names, 
                num_processes=self.cfg.test_num_processes, 
                num_episodes=self.cfg.test_num_episodes,
                record_video=self.cfg.record_video, 
                frame_stack=self.cfg.car_racing.frame_stack,
                grayscale=self.cfg.car_racing.grayscale,
                num_action_repeat=self.cfg.car_racing.num_action_repeat,
                use_global_critic=self.cfg.use_global_critic,
                use_global_policy=self.cfg.use_global_policy,
                device=self.device)

    def train(self):
        # === Train === 
        last_checkpoint_idx = getattr(self.train_runner, self.cfg.logging.checkpoint_basis)
        update_start_time = self.timer()
        num_updates = int(self.cfg.algorithm.num_env_steps) // self.cfg.algorithm.num_steps // self.cfg.algorithm.num_processes
        for j in range(self.initial_update_count, num_updates):
            stats = self.train_runner.run()

            # === Perform logging ===
            if self.train_runner.num_updates <= self.last_logged_update_at_restart:
                continue

            log = (j % self.cfg.logging.log_interval == 0) or j == num_updates - 1
            save_screenshot = \
                self.cfg.logging.screenshot_interval > 0 and \
                    (j % self.cfg.logging.screenshot_interval == 0)

            if log:
                # Eval
                test_stats = {}
                if self.evaluator is not None and (j % self.cfg.test_interval == 0 or j == num_updates - 1):
                    test_stats = self.evaluator.evaluate(self.train_runner.agents['agent'])
                    stats.update(test_stats)
                else:
                    stats.update({k:None for k in self.evaluator.get_stats_keys()})

                update_end_time = self.timer()
                num_incremental_updates = 1 if j == 0 else self.cfg.logging.log_interval
                sps = num_incremental_updates*(self.cfg.algorithm.num_processes * self.cfg.algorithm.num_steps) / (update_end_time - update_start_time)
                update_start_time = update_end_time
                stats.update({'sps': sps})
                stats.update(test_stats) # Ensures sps column is always before test stats
                self.log_stats(stats)

            checkpoint_idx = getattr(self.train_runner, self.cfg.logging.checkpoint_basis)

            if checkpoint_idx != last_checkpoint_idx:
                is_last_update = j == num_updates - 1
                if is_last_update or \
                    (self.train_runner.num_updates > 0 and checkpoint_idx % self.cfg.logging.checkpoint_interval == 0):
                    self.checkpoint(checkpoint_idx)
                    logging.info(f"\nSaved checkpoint after update {j}")
                    logging.info(f"\nLast update: {is_last_update}")
                elif self.train_runner.num_updates > 0 and self.cfg.logging.archive_interval > 0 \
                    and checkpoint_idx % self.cfg.logging.archive_interval == 0:
                    self.checkpoint(checkpoint_idx)
                    logging.info(f"\nArchived checkpoint after update {j}")

            if save_screenshot:
                level_info = self.train_runner.sampled_level_info
                if self.cfg.env_name.startswith('BipedalWalker'):
                    encodings = self.venv.get_level()
                    df = bipedalwalker_df_from_encodings(self.cfg.env_name, encodings)
                    if self.cfg.accel.use_editor and level_info:
                        df.to_csv(os.path.join(
                            self.screenshot_dir, 
                            f"update{j}-replay{level_info['level_replay']}-n_edits{level_info['num_edits'][0]}.csv"))
                    else:
                        df.to_csv(os.path.join(
                            self.screenshot_dir, 
                            f'update{j}.csv'))
                else:
                    self.venv.reset_agent()
                    images = self.venv.get_images()
                    if self.cfg.accel.use_editor and level_info:
                        save_images(
                            images[:self.cfg.logging.screenshot_batch_size], 
                            os.path.join(
                                self.screenshot_dir, 
                                f"update{j}-replay{level_info['level_replay']}-n_edits{level_info['num_edits'][0]}.png"), 
                            normalize=True, channels_first=False)
                    else:
                        save_images(
                            images[:self.cfg.logging.screenshot_batch_size], 
                            os.path.join(self.screenshot_dir, f'update{j}.png'),
                            normalize=True, channels_first=False)
                    plt.close()

        self.evaluator.close()
        self.venv.close()

        if display:
            display.stop()

    def log_stats(self, stats):
        self.filewriter.log(stats)
        if self.cfg.logging.verbose:
            HumanOutputFormat(sys.stdout).writekvs(stats)

    def checkpoint(self, index=None):
        if self.cfg.logging.disable_checkpoint:
            return
        safe_checkpoint({'runner_state_dict': self.train_runner.state_dict()}, 
                        self.checkpoint_path,
                        index=index, 
                        archive_interval=self.cfg.logging.archive_interval)
        logging.info("Saved checkpoint to %s", self.checkpoint_path)

@hydra.main(config_path='conf/.', config_name='mg_25b_dr', version_base="1.1")
def main(cfg):
    from train import Workspace as W
    workspace = W(cfg)
    workspace.train()

if __name__ == '__main__':
    main()
