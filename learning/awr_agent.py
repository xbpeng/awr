import gym
import numpy as np
import tensorflow as tf

import learning.rl_agent as rl_agent
import util.net_util as net_util
import util.rl_path as rl_path

'''
Advantage-Weighted Regression Agent
'''

class AWRAgent(rl_agent.RLAgent):
    ADV_EPS = 1e-5

    def __init__(self, 
                 env,
                 sess,

                 actor_net_layers=[128, 64],
                 actor_stepsize=0.00005,
                 actor_momentum=0.9,
                 actor_init_output_scale=0.01,
                 actor_batch_size=256,
                 actor_steps=1000,
                 action_std=0.2,
                 action_l2_weight=0.0,
                 action_entropy_weight=0.0,

                 critic_net_layers=[128, 64],
                 critic_stepsize=0.01,
                 critic_momentum=0.9,
                 critic_batch_size=256,
                 critic_steps=500,

                 discount=0.99,
                 samples_per_iter=2048,
                 replay_buffer_size=50000,
                 normalizer_samples=300000,

                 weight_clip=20,
                 td_lambda=0.95,
                 temp=1.0,

                 visualize=False):
        
        self._actor_net_layers = actor_net_layers
        self._actor_stepsize = actor_stepsize
        self._actor_momentum = actor_momentum
        self._actor_init_output_scale = actor_init_output_scale
        self._actor_batch_size = actor_batch_size
        self._actor_steps = actor_steps
        self._action_std = action_std
        self._action_l2_weight = action_l2_weight
        self._action_entropy_weight = action_entropy_weight

        self._critic_net_layers = critic_net_layers
        self._critic_stepsize = critic_stepsize
        self._critic_momentum = critic_momentum
        self._critic_batch_size = critic_batch_size
        self._critic_steps = critic_steps

        self._weight_clip = weight_clip
        self._td_lambda = td_lambda
        self._temp = temp
        
        self._critic_step_count = 0
        self._actor_steps_count = 0

        self._actor_bound_loss_weight = 10.0

        super().__init__(env=env,
                         sess=sess,
                         discount=discount,
                         samples_per_iter=samples_per_iter,
                         replay_buffer_size=replay_buffer_size,
                         normalizer_samples=normalizer_samples,
                         visualize=visualize)
        return

    def sample_action(self, s, test):
        n = len(s.shape)
        s = np.reshape(s, [-1, self.get_state_size()])

        feed = {
            self._s_tf : s
        }

        if (test):
            run_tfs = [self._mode_a_tf, self._mode_a_logp_tf]
        else:
            run_tfs = [self._sample_a_tf, self._sample_a_logp_tf]

        a, logp = self._sess.run(run_tfs, feed_dict=feed)

        if n == 1:
            a = a[0]
            logp = logp[0]
        return a, logp

    def eval_critic(self, s):
        n = len(s.shape)
        s = np.reshape(s, [-1, self.get_state_size()])

        feed = {
            self._s_tf : s
        }
        v = self._sess.run(self._critic_tf, feed_dict=feed)

        if n == 1:
            v = v[0]
        return v

    def train(self, max_iter, test_episodes, output_dir, output_iters):
        self._critic_step_count = 0
        self._actor_step_count = 0
        super().train(max_iter=max_iter, 
                      test_episodes=test_episodes,
                      output_dir=output_dir, 
                      output_iters=output_iters)
        return

    def _build_nets(self):
        s_size = self.get_state_size()
        a_size = self.get_action_size()
        action_space = self.get_action_space()

        self._s_tf = tf.placeholder(tf.float32, shape=[None, s_size], name="s")
        self._a_tf = tf.placeholder(tf.float32, shape=[None, a_size], name="a")
        self._tar_val_tf = tf.placeholder(tf.float32, shape=[None], name="tar_val")
        self._a_w_tf = tf.placeholder(tf.float32, shape=[None], name="a_w")

        with tf.variable_scope(self.MAIN_SCOPE):
            with tf.variable_scope(self.ACTOR_SCOPE):
                self._norm_a_pd_tf = self._build_net_actor(self._get_actor_inputs())

            with tf.variable_scope(self.CRITIC_SCOPE):
                self._norm_critic_tf = self._build_net_critic(self._get_critic_inputs())
                self._critic_tf = self._val_norm.unnormalize_tf(self._norm_critic_tf)
        
        sample_norm_a_tf = self._norm_a_pd_tf.sample()
        self._sample_a_logp_tf = self._norm_a_pd_tf.log_prob(sample_norm_a_tf)
        self._sample_a_tf = self._a_norm.unnormalize_tf(tf.cast(sample_norm_a_tf, tf.float32))
        if (len(self._sample_a_tf.shape) == 1):
            self._sample_a_tf = tf.expand_dims(self._sample_a_tf, axis=-1)

        mode_norm_a_tf = self._norm_a_pd_tf.mode()
        self._mode_a_logp_tf = self._norm_a_pd_tf.log_prob(mode_norm_a_tf)
        self._mode_a_tf = self._a_norm.unnormalize_tf(tf.cast(mode_norm_a_tf, tf.float32))
        if (len(self._mode_a_tf.shape) == 1):
            self._mode_a_tf = tf.expand_dims(self._mode_a_tf, axis=-1)

        norm_a_tf = self._a_norm.normalize_tf(self._a_tf)
        if (isinstance(action_space, gym.spaces.Discrete)):
            norm_a_tf = tf.squeeze(norm_a_tf, axis=-1)
            norm_a_tf = tf.cast(norm_a_tf, tf.int32)
        self._a_logp_tf = self._norm_a_pd_tf.log_prob(norm_a_tf)
        return

    def _build_losses(self):
        norm_tar_val_tf = self._val_norm.normalize_tf(self._tar_val_tf)
        norm_val_diff = norm_tar_val_tf - self._norm_critic_tf
        self._critic_loss_tf = 0.5 * tf.reduce_mean(tf.square(norm_val_diff))

        self._actor_loss_tf = self._a_w_tf * self._a_logp_tf
        self._actor_loss_tf = -tf.reduce_mean(self._actor_loss_tf)

        self._actor_loss_tf += self._actor_bound_loss_weight * self._action_bound_loss(self._norm_a_pd_tf)

        if (self._action_l2_weight != 0):
            self._actor_loss_tf += self._action_l2_weight * self._action_l2_loss(self._norm_a_pd_tf)

        if (self._action_entropy_weight != 0):
            self._actor_loss_tf += self._action_entropy_weight * self._action_entropy_loss(self._norm_a_pd_tf)

        return

    def _build_solvers(self):
        critic_vars = self._tf_vars(self.MAIN_SCOPE + "/" + self.CRITIC_SCOPE)
        self._critic_opt = tf.train.MomentumOptimizer(learning_rate=self._critic_stepsize, momentum=self._critic_momentum)
        self._update_critic_op = self._critic_opt.minimize(self._critic_loss_tf, var_list=critic_vars)

        actor_vars = self._tf_vars(self.MAIN_SCOPE + "/" + self.ACTOR_SCOPE)
        self._actor_opt = tf.train.MomentumOptimizer(learning_rate=self._actor_stepsize, momentum=self._actor_momentum)
        self._update_actor_op = self._actor_opt.minimize(self._actor_loss_tf, var_list=actor_vars)
        return

    def _get_actor_inputs(self):
        norm_s_tf = self._s_norm.normalize_tf(self._s_tf)
        input_tfs = [norm_s_tf]
        return input_tfs

    def _get_critic_inputs(self):
        norm_s_tf = self._s_norm.normalize_tf(self._s_tf)
        input_tfs = [norm_s_tf]
        return input_tfs

    def _build_net_actor(self, input_tfs, reuse=False):
        h = net_util.build_fc_net(input_tfs=input_tfs, layers=self._actor_net_layers, reuse=reuse)
        norm_a_pd_tf = self._build_action_pd(input_tf=h, init_output_scale=self._actor_init_output_scale,
                                             reuse=reuse)
        return norm_a_pd_tf

    def _build_net_critic(self, input_tfs, reuse=False):
        out_size = 1
        h = net_util.build_fc_net(input_tfs=input_tfs, layers=self._critic_net_layers, reuse=reuse)
        norm_val_tf = tf.layers.dense(inputs=h, units=out_size, activation=None,
                                kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                reuse=reuse);
        norm_val_tf = tf.squeeze(norm_val_tf, axis=-1)
        return norm_val_tf
    
    def _update(self, iter, new_sample_count):
        idx = np.array(self._replay_buffer.get_unrolled_indices())
        
        end_mask = self._replay_buffer.is_path_end(idx)
        valid_mask = np.logical_not(end_mask)
        valid_idx = idx[valid_mask]
        valid_idx = np.column_stack([valid_idx, np.nonzero(valid_mask)[0]])
        
        # update critic
        vals = self._compute_batch_vals(idx)
        new_vals = self._compute_batch_new_vals(idx, vals)
        
        critic_steps = int(np.ceil(self._critic_steps * new_sample_count / self._samples_per_iter))
        critic_losses = self._update_critic(critic_steps, valid_idx, new_vals)

        # update actor
        vals = self._compute_batch_vals(idx)
        new_vals = self._compute_batch_new_vals(idx, vals)
        adv, norm_adv, adv_mean, adv_std = self._calc_adv(new_vals, vals, valid_mask)
        adv_weights, adv_weights_mean, adv_weights_min, adv_weights_max = self._calc_adv_weights(norm_adv, valid_mask)
        
        actor_steps = int(np.ceil(self._actor_steps * new_sample_count / self._samples_per_iter))
        actor_losses = self._update_actor(actor_steps, valid_idx, adv_weights)


        self._critic_step_count += critic_steps
        self._actor_step_count += actor_steps
        
        self._logger.log_tabular("Critic_Loss", critic_losses["loss"])
        self._logger.log_tabular("Critic_Steps", self._critic_step_count)
        self._logger.log_tabular("Actor_Loss", actor_losses["loss"])
        self._logger.log_tabular("Actor_Steps", self._actor_step_count)
        
        self._logger.log_tabular("Adv_Mean", adv_mean)
        self._logger.log_tabular("Adv_Std", adv_std)
        self._logger.log_tabular("Adv_Weights_Min", adv_weights_min)
        self._logger.log_tabular("Adv_Weights_Mean", adv_weights_mean)
        self._logger.log_tabular("Adv_Weights_Max", adv_weights_max)
        
        info = {"critic_info": critic_losses, "actor_info": actor_losses}
        return info
    
    def _update_critic(self, steps, sample_idx, tar_vals):
        num_idx = sample_idx.shape[0]
        steps_per_shuffle = int(np.ceil(num_idx / self._critic_batch_size))
        losses = None

        for b in range(steps):
            if (b % steps_per_shuffle == 0):
                np.random.shuffle(sample_idx)

            batch_idx_beg = b * self._critic_batch_size
            batch_idx_end = batch_idx_beg + self._critic_batch_size
            critic_batch_idx = np.array(range(batch_idx_beg, batch_idx_end), dtype=np.int32)
            critic_batch_idx = np.mod(critic_batch_idx, num_idx)

            critic_batch = sample_idx[critic_batch_idx]
            critic_batch_vals = tar_vals[critic_batch[:,1]]
            critic_s = self._replay_buffer.get("states", critic_batch[:,0])

            curr_losses = self._step_critic(critic_s, critic_batch_vals)

            if (losses is None):
                losses = curr_losses
            else:
                for key, val in curr_losses.items():
                    losses[key] += val
        
        for key in losses.keys():
            losses[key] /= steps

        return losses

    def _update_actor(self, steps, sample_idx, adv_weights):
        num_idx = sample_idx.shape[0]
        steps_per_shuffle = int(np.ceil(num_idx / self._actor_batch_size))
        losses = None

        for b in range(steps):
            if (b % steps_per_shuffle == 0):
                np.random.shuffle(sample_idx)

            batch_idx_beg = b * self._actor_batch_size
            batch_idx_end = batch_idx_beg + self._actor_batch_size
            actor_batch_idx = np.array(range(batch_idx_beg, batch_idx_end), dtype=np.int32)
            actor_batch_idx = np.mod(actor_batch_idx, num_idx)
                
            actor_batch = sample_idx[actor_batch_idx]
            actor_batch_adv = adv_weights[actor_batch[:,1]]
            actor_s = self._replay_buffer.get("states", actor_batch[:,0])
            actor_a = self._replay_buffer.get("actions", actor_batch[:,0])

            curr_losses = self._step_actor(actor_s, actor_a, actor_batch_adv)

            if (losses is None):
                losses = curr_losses
            else:
                for key, val in curr_losses.items():
                    losses[key] += val
        
        for key in losses.keys():
            losses[key] /= steps

        return losses

    def _step_critic(self, s, tar_vals):
        feed = {
            self._s_tf: s,
            self._tar_val_tf: tar_vals
        }

        run_tfs = [self._update_critic_op, self._critic_loss_tf]
        losses = self._sess.run(run_tfs, feed)
        losses = {"loss": losses[1]}
        return losses
    
    def _step_actor(self, s, a, a_w):
        feed = {
            self._s_tf: s,
            self._a_tf: a,
            self._a_w_tf: a_w,
        }

        run_tfs = [self._update_actor_op, self._actor_loss_tf]
        losses = self._sess.run(run_tfs, feed)
        losses = {"loss": losses[1]}
        return losses
    
    def _compute_batch_vals(self, idx):
        states = self._replay_buffer.get("states", idx)
        vals = self.eval_critic(states)

        is_end = self._replay_buffer.is_path_end(idx)
        is_fail = self._replay_buffer.check_terminal_flag(idx, rl_path.Terminate.Fail)
        is_fail = np.logical_and(is_end, is_fail) 

        vals[is_fail] = 0.0

        return vals

    def _compute_batch_new_vals(self, idx, val_buffer):
        # use td-lambda to compute new values
        new_vals = np.zeros_like(val_buffer)
        n = len(idx)

        start_i = 0
        while start_i < n:
            start_idx = idx[start_i]
            path_len = self._replay_buffer.get_pathlen(start_idx)
            end_i = start_i + path_len
            end_idx = idx[end_i]

            test_start_idx = self._replay_buffer.get_path_start(start_idx)
            test_end_idx = self._replay_buffer.get_path_end(start_idx)
            assert(start_idx == test_start_idx)
            assert(end_idx == test_end_idx)

            path_indices = idx[start_i:(end_i + 1)]
            r = self._replay_buffer.get("rewards", path_indices[:-1])
            v = val_buffer[start_i:(end_i + 1)]

            new_vals[start_i:end_i] = self._compute_return(r, self._discount, self._td_lambda, v)
            start_i = end_i + 1
        
        return new_vals

    def _compute_return(self, rewards, discount, td_lambda, val_t):
        # computes td-lambda return of path
        path_len = len(rewards)
        assert len(val_t) == path_len + 1

        return_t = np.zeros(path_len)
        last_val = rewards[-1] + discount * val_t[-1]
        return_t[-1] = last_val

        for i in reversed(range(0, path_len - 1)):
            curr_r = rewards[i]
            next_ret = return_t[i + 1]
            curr_val = curr_r + discount * ((1.0 - td_lambda) * val_t[i + 1] + td_lambda * next_ret)
            return_t[i] = curr_val
    
        return return_t

    def _calc_adv(self, new_vals, vals, valid_mask):
        adv = new_vals - vals

        valid_adv = adv[valid_mask]
        adv_mean = np.mean(valid_adv)
        adv_std = np.std(valid_adv)

        norm_adv = (adv - adv_mean) / (adv_std + self.ADV_EPS)
        return adv, norm_adv, adv_mean, adv_std

    def _calc_adv_weights(self, adv, valid_mask):
        weights = np.exp(adv / self._temp)

        valid_weights = weights[valid_mask]
        weights_mean = np.mean(valid_weights)
        weights_min = np.min(valid_weights)
        weights_max = np.max(valid_weights)

        weights = np.minimum(weights, self._weight_clip)
        return weights, weights_mean, weights_min, weights_max
