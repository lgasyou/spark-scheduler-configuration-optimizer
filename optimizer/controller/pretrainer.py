import logging
import os
import time

from optimizer.agent import Agent
from optimizer.environment import PreTrainEnv, StateInvalidException
from optimizer.hyperparameters import PRE_TRAIN_LOOP_INTERNAL
from optimizer.replaymemory import ReplayMemoryProxy
from optimizer.replaymemory.memoryserializer import MemorySerializer
from optimizer.util import sparkutil


class PreTrainer(object):

    TMP_MEMORY_FILENAME = './results/pre-train-memory.tmp.pk'

    def __init__(self, args, memory: ReplayMemoryProxy, agent: Agent,
                 action_space: int, priority_weight_increase):
        self.logger = logging.getLogger(__name__)
        self.args = args
        self.mem = memory
        self.memory_serializer = MemorySerializer(memory)
        self.agent = agent
        self.env = PreTrainEnv(self.args)

        self.priority_weight_increase = priority_weight_increase
        self.action_space = action_space
        self.t = 0

    def start_from_breakpoint(self):
        self.logger.info('Pre-training DQN model...')

        if not self.memory_serializer.try_load():
            self.memory_serializer.try_load_by_filename(self.TMP_MEMORY_FILENAME)
            self.t = self.mem.t
            self.logger.info(self.t)
            while True:
                self._train_step()

        # Pre-train DQN model by using training set
        self.agent.learn(self.mem)
        self.logger.info('Pre-training DQN model finished.')

    def _train_step(self):
        pre_train_set = self.env.generate_pre_train_set()
        for action_index in range(self.action_space):
            self.env.start(action_index, pre_train_set)
            self._train_once(action_index)
            self.memory_serializer.save_as(self.TMP_MEMORY_FILENAME)
            sparkutil.clean_spark_log(os.getcwd(), self.args.hadoop_home)

    def _train_once(self, action_index: int):
        done = False
        while not done:
            try:
                state, reward, done = self.env.step()
                self.mem.append(state, action_index, reward, done)
                self.t += 1
                self.logger.info('Iteration: %d, Action: %d, Reward: %f' % (self.t, action_index, reward))

                # Train and test
                if self.t >= self.args.learn_start:
                    self._agent_learn()

            except StateInvalidException:
                pass
            time.sleep(PRE_TRAIN_LOOP_INTERNAL)

        self.mem.terminate()

    def _agent_learn(self):
        mem, agent, env = self.mem, self.agent, self.env
        replay_frequency = self.args.replay_frequency
        target_update = self.args.target_update

        # Anneal importance sampling weight β to 1
        mem.priority_weight = min(mem.priority_weight + self.priority_weight_increase, 1)

        if self.t % replay_frequency == 0:
            agent.learn(mem)  # Train with n-step distributional double-Q learning

        # Update target network
        if self.t % target_update == 0:
            agent.update_target_net()
