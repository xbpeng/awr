import abc
import copy
import gym
import numpy as np
import os
import tensorflow as tf
import time

import util.logger as logger
import util.normalizer as normalizer
import util.replay_buffer as replay_buffer
import util.rl_path as rl_path

'''
Reinforcement Learning Agent
'''

class RLAgent(abc.ABC):
    MAIN_SCOPE = "main"
    ACTOR_SCOPE = "actor"
    CRITIC_SCOPE = "critic"
    SOLVER_SCOPE = "solver"
    RESOURCE_SCOPE = "resource"

    def __init__(self, 
                 env,
                 sess,
                 discount=0.99,
                 samples_per_iter=2048,
                 replay_buffer_size=50000,
                 normalizer_samples=100000,
                 visualize=False):

        self._env = env
        self._sess = sess

        self._discount = discount
        self._samples_per_iter = samples_per_iter
        self._normalizer_samples = normalizer_samples
        self._replay_buffer = self._build_replay_buffer(replay_buffer_size)
        
        self.visualize = visualize
        
        self._logger = None

        with self._sess.as_default(), self._sess.graph.as_default():
            with tf.variable_scope(self.RESOURCE_SCOPE):
                self._build_normalizers()

            self._build_nets()

            with tf.variable_scope(self.SOLVER_SCOPE):
                self._build_losses()
                self._build_solvers()

            self._init_vars()
            self._build_saver()

        return

    def get_state_size(self):
        state_size = np.prod(self._env.observation_space.shape)
        return state_size

    def get_action_size(self):
        action_size = 0
        action_space = self.get_action_space()

        if (isinstance(action_space, gym.spaces.Box)):
            action_size = np.prod(action_space.shape)
        elif (isinstance(action_space, gym.spaces.Discrete)):
            action_size = 1
        else:
            assert False, "Unsupported action space: " + str(self._env.action_space)

        return action_size

    def get_action_space(self):
        return self._env.action_space

    def get_total_samples(self):
        return self._replay_buffer.total_count

    def eval(self, num_episodes):
        test_return, test_path_count = self._rollout_test(num_episodes, print_info=True)
        logger.Logger.print("Test_Return: {:.3f}".format(test_return))
        logger.Logger.print("Test_Paths: {:.3f}".format(test_path_count))
        return

    def train(self, max_iter, test_episodes, output_dir, output_iters):
        log_file = os.path.join(output_dir, "log.txt")
        self._logger = logger.Logger()
        self._logger.configure_output_file(log_file)
        
        model_file = os.path.join(output_dir, "model.ckpt")

        iter = 0
        total_train_return = 0
        total_train_path_count = 0
        test_return = 0
        test_path_count = 0
        start_time = time.time()

        while (iter < max_iter):
            train_return, train_path_count, new_sample_count = self._rollout_train(self._samples_per_iter)

            total_train_return += train_path_count * train_return
            total_train_path_count += train_path_count
            avg_train_return = total_train_return / total_train_path_count
            
            total_samples = self.get_total_samples()
            wall_time = time.time() - start_time
            wall_time /= 60 * 60 # store time in hours
            
            self._logger.log_tabular("Iteration", iter)
            self._logger.log_tabular("Wall_Time", wall_time)
            self._logger.log_tabular("Samples", total_samples)
            self._logger.log_tabular("Train_Return", avg_train_return)
            self._logger.log_tabular("Train_Paths", total_train_path_count)
            self._logger.log_tabular("Test_Return", test_return)
            self._logger.log_tabular("Test_Paths", test_path_count)
            
            if (self._need_normalizer_update() and iter == 0):
                self._update_normalizers()

            self._update(iter, new_sample_count)
            
            if (self._need_normalizer_update()):
                self._update_normalizers()

            if (iter % output_iters == 0):
                test_return, test_path_count = self._rollout_test(test_episodes, print_info=False)
                self._logger.log_tabular("Test_Return", test_return)
                self._logger.log_tabular("Test_Paths", test_path_count)

                self.save_model(model_file)
                self._logger.print_tabular()
                self._logger.dump_tabular()
                
                total_train_return = 0
                total_train_path_count = 0
            else:
                self._logger.print_tabular()

            iter += 1

        return
    
    def save_model(self, out_path):
        try:
            save_path = self._saver.save(self._sess, out_path, write_meta_graph=False, write_state=False)
            logger.Logger.print("Model saved to: " + save_path)
        except:
            logger.Logger.print("Failed to save model to: " + out_path)
        return

    def load_model(self, in_path):
        self._saver.restore(self._sess, in_path)
        self._load_normalizers()
        logger.Logger.print("Model loaded from: " + in_path)
        return

    def get_state_bound_min(self):
        return self._env.observation_space.low
    
    def get_state_bound_max(self):
        return self._env.observation_space.high

    def get_action_bound_min(self):
        action_space = self.get_action_space()
        if (isinstance(action_space, gym.spaces.Box)):
            bound_min = self._env.action_space.low
        else:
            bound_min = -np.inf * np.ones(1)
        return bound_min

    def get_action_bound_max(self):
        action_space = self.get_action_space()
        if (isinstance(action_space, gym.spaces.Box)):
            bound_max = self._env.action_space.high
        else:
            bound_max = np.inf * np.ones(1)
        return bound_max

    def render_env(self):
        self._env.render()
        return

    def _build_normalizers(self):
        self._s_norm = self._build_normalizer_state()
        self._a_norm = self._build_normalizer_action()
        self._val_norm = self._build_normalizer_val()
        return

    def _need_normalizer_update(self):
        return self._s_norm.need_update()

    def _build_normalizer_state(self):
        size = self.get_state_size()

        high = self.get_state_bound_max().copy()
        low = self.get_state_bound_min().copy()
        inf_mask = np.logical_or((high >= np.finfo(np.float32).max), (low <= np.finfo(np.float32).min))
        high[inf_mask] = 1.0
        low[inf_mask] = -1.0
        
        mean = 0.5 * (high + low)
        std = 0.5 * (high - low)

        norm = normalizer.Normalizer(sess=self._sess, scope="s_norm", size=size, init_mean=mean, init_std=std)

        return norm

    def _build_normalizer_action(self):
        size = self.get_action_size()

        high = self.get_action_bound_max().copy()
        low = self.get_action_bound_min().copy()
        inf_mask = np.logical_or((high >= np.finfo(np.float32).max), (low <= np.finfo(np.float32).min))
        high[inf_mask] = 1.0
        low[inf_mask] = -1.0
        
        mean = 0.5 * (high + low)
        std = 0.5 * (high - low)

        norm = normalizer.Normalizer(sess=self._sess, scope="a_norm", size=size, init_mean=mean, init_std=std)

        return norm

    def _build_normalizer_val(self):
        mean = 0.0
        std = 1.0 / (1.0 - self._discount)
        norm = normalizer.Normalizer(sess=self._sess, scope="val_norm", size=1, init_mean=mean, init_std=std)
        return norm

    def _build_replay_buffer(self, buffer_size):
        buffer = replay_buffer.ReplayBuffer(buffer_size=buffer_size)
        return buffer
    
    @abc.abstractmethod
    def sample_action(self, s, test):
        pass

    @abc.abstractmethod
    def _build_nets(self):
        pass

    @abc.abstractmethod
    def _build_losses(self):
        pass
    
    @abc.abstractmethod
    def _build_solvers(self):
        pass

    @abc.abstractmethod
    def _update(self, iter, new_sample_count):
        pass

    def _init_vars(self):
        self._sess.run(tf.global_variables_initializer())
        return

    def _build_saver(self):
        vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        vars = [v for v in vars if self.SOLVER_SCOPE + '/' not in v.name]
        assert len(vars) > 0
        self._saver = tf.train.Saver(vars, max_to_keep=0)
        return
    
    def _rollout_train(self, num_samples):
        new_sample_count = 0
        total_return = 0
        path_count = 0

        while (new_sample_count < num_samples):
            path = self._rollout_path(test=False)
            path_id = self._replay_buffer.store(path)
            valid_path = path_id != replay_buffer.INVALID_IDX

            if valid_path:
                path_return = path.calc_return()

                if (self._enable_normalizer_update()):
                    self._record_normalizers(path)

                new_sample_count += path.pathlength()
                total_return += path_return
                path_count += 1
            else:
                assert False, "Invalid path detected"

        avg_return = total_return / path_count

        return avg_return, path_count, new_sample_count

    def _rollout_test(self, num_episodes, print_info=False):
        total_return = 0
        for e in range(num_episodes):
            path = self._rollout_path(test=True)
            path_return = path.calc_return()
            total_return += path_return

            if (print_info):
                logger.Logger.print("Episode: {:d}".format(e))
                logger.Logger.print("Curr_Return: {:.3f}".format(path_return))
                logger.Logger.print("Avg_Return: {:.3f}\n".format(total_return / (e + 1)))

        avg_return = total_return / num_episodes
        return avg_return, num_episodes

    def _rollout_path(self, test):
        path = rl_path.RLPath()

        s = self._env.reset()
        s = np.array(s)
        path.states.append(s)

        done = False
        while not done:
            a, logp = self.sample_action(s, test)
            s, r, done, info = self._step_env(a)
            s = np.array(s)
            
            path.states.append(s)
            path.actions.append(a)
            path.rewards.append(r)
            path.logps.append(logp)

            if (self.visualize):
                self.render_env()

        path.terminate = self._check_env_termination()

        return path

    def _step_env(self, a):
        if (isinstance(self._env.action_space, gym.spaces.Discrete)):
            a = int(a[0])
        output = self._env.step(a)
        return output

    def _check_env_termination(self):
        if (self._env._elapsed_steps >= self._env._max_episode_steps):
           term = rl_path.Terminate.Null
        else:
           term = rl_path.Terminate.Fail
        return term

    def _record_normalizers(self, path):
        states = np.array(path.states)
        self._s_norm.record(states)
        return

    def _update_normalizers(self):
        self._s_norm.update()
        return

    def _load_normalizers(self):
        self._s_norm.load()
        self._a_norm.load()
        self._val_norm.load()
        return

    def _build_action_pd(self, input_tf, init_output_scale, reuse=False):
        action_space = self.get_action_space()

        if (isinstance(action_space, gym.spaces.Box)):
            output_size = self.get_action_size()

            mean_kernel_init = tf.random_uniform_initializer(minval=-init_output_scale, maxval=init_output_scale)
            mean_bias_init = tf.zeros_initializer()
            logstd_kernel_init = tf.random_uniform_initializer(minval=-init_output_scale, maxval=init_output_scale)
            logstd_bias_init = np.log(self._action_std) * np.ones(output_size)
            logstd_bias_init = logstd_bias_init.astype(np.float32)

            with tf.variable_scope("mean", reuse=reuse):
                mean_tf = tf.layers.dense(inputs=input_tf, units=output_size,
                                            kernel_initializer=mean_kernel_init,
                                            bias_initializer=mean_bias_init,
                                            activation=None)
            with tf.variable_scope("logstd", reuse=reuse):
                logstd_tf = tf.get_variable(dtype=tf.float32, name="bias", initializer=logstd_bias_init,
                                            trainable=False)
                logstd_tf = tf.broadcast_to(logstd_tf, tf.shape(mean_tf))
                std_tf = tf.exp(logstd_tf)

            a_pd_tf = tf.contrib.distributions.MultivariateNormalDiag(loc=mean_tf, scale_diag=std_tf)

        elif (isinstance(action_space, gym.spaces.Discrete)):
            output_size = self._env.action_space.n
            
            kernel_init = tf.random_uniform_initializer(minval=-init_output_scale, maxval=init_output_scale)
            bias_init = tf.zeros_initializer()

            with tf.variable_scope("logits", reuse=reuse):
                logits_tf = tf.layers.dense(inputs=input_tf, units=output_size,
                                            kernel_initializer=kernel_init,
                                            bias_initializer=bias_init,
                                            activation=None)
            a_pd_tf = tf.contrib.distributions.Categorical(logits=logits_tf)
            
        else:
            assert False, "Unsupported action space: " + str(self._env.action_space)

        return a_pd_tf

    def _tf_vars(self, scope=""):
        vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)
        assert len(vars) > 0
        return vars

    def _enable_normalizer_update(self):
        sample_count = self.get_total_samples()
        enable_update = sample_count < self._normalizer_samples
        return enable_update
    
    def _action_bound_loss(self, a_pd_tf):
        action_space = self.get_action_space()
        if (isinstance(action_space, gym.spaces.Box)):
            axis = -1
            a_bound_min = self.get_action_bound_min()
            a_bound_max = self.get_action_bound_max()
            assert(np.all(np.isfinite(a_bound_min)) and np.all(np.isfinite(a_bound_max))), "Actions must be bounded."

            norm_a_bound_min = self._a_norm.normalize(a_bound_min)
            norm_a_bound_max = self._a_norm.normalize(a_bound_max)

            val = a_pd_tf.mean()
            violation_min = tf.minimum(val - norm_a_bound_min, 0)
            violation_max = tf.maximum(val - norm_a_bound_max, 0)
            violation = tf.reduce_sum(tf.square(violation_min), axis=axis) \
                        + tf.reduce_sum(tf.square(violation_max), axis=axis)

            a_bound_loss = 0.5 * tf.reduce_mean(violation)
        else:
            a_bound_loss = tf.zeros(shape=[])

        return a_bound_loss

    def _action_l2_loss(self, a_pd_tf):
        action_space = self.get_action_space()
        if (isinstance(action_space, gym.spaces.Box)):
            val = a_pd_tf.mean()
        elif (isinstance(action_space, gym.spaces.Discrete)):
            val = a_pd_tf.logits
        else:
            assert False, "Unsupported action space: " + str(self._env.action_space)

        loss = tf.reduce_sum(tf.square(val), axis=-1)
        loss = 0.5 * tf.reduce_mean(loss)
        return loss

    def _action_entropy_loss(self, a_pd_tf):
        loss = a_pd_tf.entropy()
        loss = -tf.reduce_mean(loss)
        return loss
