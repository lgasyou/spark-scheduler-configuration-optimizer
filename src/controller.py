import argparse
from datetime import datetime
from typing import Tuple

from tqdm import tqdm

from .agent import Agent
from .env import Env, GoogleTraceEnv
from .memory import ReplayMemory
from .test import test


# Simple ISO 8601 timestamped logger
def log(s):
    print('[' + str(datetime.now().strftime('%Y-%m-%dT%H:%M:%S')) + '] ' + s)


class OptimizationController(object):
    """
    Cluster Scheduler Configuration Optimizer
    -----------------------------------------
    算法流程：
    1.	用随机权重初始化价值函数；
    2.	预训练DQN模型；
    3.	f设置为0；
    4.	for 片段(episode) 1 to
    5.	    for 时间片 1 to
    6.	       	在状态下用贪婪算法选一个动作；
    7.	 		执行动作并让调度器去观察奖励和下一个状态
    8.	 		把样本加到中；
    9.	 		if (f % F == 0)
    10.	 			使用中的样本训练深度学习模型DQN；
    11.	 		end if
    12.         f = f + 1
    13.	 	end for
    14.	end for
    """

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.env, self.action_space = self.__setup_env()
        self.train_env = GoogleTraceEnv(args)
        self.dqn, self.mem, self.priority_weight_increase = self.__setup_agent()
        self.val_mem = self.__setup_val_mem()

    # Setup Environment
    def __setup_env(self) -> Tuple[Env, int]:
        env = Env(self.args)
        action_space = env.action_space()
        return env, action_space

    # Setup Agent
    def __setup_agent(self) -> Tuple[Agent, ReplayMemory, float]:
        dqn = Agent(self.args, self.env)
        mem = ReplayMemory(self.args, self.args.memory_capacity)
        priority_weight_increase = (1 - self.args.priority_weight) / (self.args.T_max - self.args.learn_start)
        return dqn, mem, priority_weight_increase

    # Construct validation memory
    def __setup_val_mem(self) -> ReplayMemory:
        val_mem = ReplayMemory(self.args, self.args.evaluation_size)
        # T, done = 0, True
        # while T < self.args.evaluation_size:
        #     if done:
        #         state, done = self.env.reset(), False
        #
        #     next_state, _, done = self.env.step(random.randint(0, self.action_space - 1))
        #     val_mem.append(state, None, None, done)
        #     state = next_state
        #     T += 1
        return val_mem

    # Pre-train DQN model with offline data
    def pre_train_model(self):
        # Try to load data from file, if fails run training set and
        # save them into memory.
        if not self.mem.try_load_from_file():
            # Get data generator
            generator = self.train_env.get_generator()

            # Generate data, then save them into self.mem
            for step in generator:
                for (state, action, reward, terminal) in step:
                    self.mem.append(state, action, reward, terminal)

            # Save data as 'pre-train-replay-memory.pk'
            self.mem.save()

        # Pre-train DQN model by using training set
        self.dqn.learn(self.mem)

    # Start to optimize YARN scheduler
    def run(self) -> None:
        num_training_steps = self.args.T_max
        reward_clip = self.args.reward_clip

        T, done = 0, True
        self.dqn.train()
        for T in tqdm(range(num_training_steps)):
            if done:
                state, done = self.env.reset(), False

            if T % self.args.replay_frequency == 0:
                self.dqn.reset_noise()  # Draw a new set of noisy weights

            action = self.dqn.act(state)  # Choose an action greedily (with noisy weights)
            next_state, reward, done = self.env.step(action)  # Step
            if reward_clip > 0:
                reward = max(min(reward, reward_clip), -reward_clip)  # Clip rewards
            self.mem.append(state, action, reward, done)  # Append transition to memory
            T += 1

            # Train and test
            if T >= self.args.learn_start:
                # Anneal importance sampling weight β to 1
                self.mem.priority_weight = min(self.mem.priority_weight + self.priority_weight_increase, 1)

                if T % self.args.replay_frequency == 0:
                    self.dqn.learn(self.mem)  # Train with n-step distributional double-Q learning

                if T % self.args.evaluation_interval == 0:
                    self.dqn.eval()  # Set DQN (online network) to evaluation mode
                    avg_reward, avg_Q = test(self.args, T, self.dqn, self.val_mem)  # Test
                    log('T = ' + str(T) + ' / ' + str(num_training_steps) + ' | Avg. reward: ' +
                        str(avg_reward) + ' | Avg. Q: ' + str(avg_Q))
                    self.dqn.train()  # Set DQN (online network) back to training mode

                # Update target network
                if T % self.args.target_update == 0:
                    self.dqn.update_target_net()
            state = next_state

        self.env.close()
