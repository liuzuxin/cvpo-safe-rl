from copy import deepcopy

import gym
import numpy as np
import torch
import torch.nn as nn
from scipy.optimize import minimize
from safe_rl.policy.base_policy import Policy
from safe_rl.policy.model.mlp_ac import (EnsembleQCritic, CholeskyGaussianActor)
from safe_rl.util.logger import EpochLogger
from safe_rl.util.torch_util import (count_vars, get_device_name, to_device, to_ndarray,
                                     to_tensor)
from torch.distributions.uniform import Uniform
from torch.distributions import MultivariateNormal
from torch.optim import Adam
from torch.nn.utils import clip_grad_norm_


def bt(m: torch.tensor):
    return m.transpose(dim0=-2, dim1=-1)


def btr(m: torch.tensor):
    return m.diagonal(dim1=-2, dim2=-1).sum(-1)


def safe_inverse(A, det):
    indices = torch.where(det <= 1e-6)
    # pseudoinverse
    if len(indices[0]) > 0:
        return torch.linalg.pinv(A)
    return A.inverse()


def gaussian_kl(μi, μ, Ai, A):
    """
    decoupled KL between two multivariate gaussian distribution
    C_μ = KL(f(x|μi,Σi)||f(x|μ,Σi))
    C_Σ = KL(f(x|μi,Σi)||f(x|μi,Σ))
    :param μi: (B, n)
    :param μ: (B, n)
    :param Ai: (B, n, n)
    :param A: (B, n, n)
    :return: C_μ, C_Σ: scalar
        mean and covariance terms of the KL
    :return: mean of determinanats of Σi, Σ
    ref : https://stanford.edu/~jduchi/projects/general_notes.pdf page.13
    """
    n = A.size(-1)
    μi = μi.unsqueeze(-1)  # (B, n, 1)
    μ = μ.unsqueeze(-1)  # (B, n, 1)
    Σi = Ai @ bt(Ai)  # (B, n, n)
    Σ = A @ bt(A)  # (B, n, n)
    Σi_det = Σi.det()  # (B,)
    Σ_det = Σ.det()  # (B,)
    Σi_inv = safe_inverse(Σi, Σi_det)  # (B, n, n)
    Σ_inv = safe_inverse(Σ, Σ_det)  # (B, n, n)
    # determinant can be minus due to numerical calculation error
    # https://github.com/daisatojp/mpo/issues/11
    Σi_det = torch.clamp_min(Σi_det, 1e-6)
    Σ_det = torch.clamp_min(Σ_det, 1e-6)
    inner_μ = ((μ - μi).transpose(-2, -1) @ Σi_inv @ (μ - μi)).squeeze()  # (B,)
    inner_Σ = torch.log(Σ_det / Σi_det) - n + btr(Σ_inv @ Σi)  # (B,)
    C_μ = 0.5 * torch.mean(inner_μ)
    C_Σ = 0.5 * torch.mean(inner_Σ)
    return C_μ, C_Σ, torch.mean(Σi_det), torch.mean(Σ_det)


class CVPO(Policy):
    def __init__(self,
                 env: gym.Env,
                 logger: EpochLogger,
                 num_qc=1,
                 cost_limit=40,
                 use_cost_decay=False,
                 cost_start=100,
                 cost_end=10,
                 decay_epoch=100,
                 timeout_steps=200,
                 dual_constraint=0.1,
                 kl_mean_constraint=0.01,
                 kl_var_constraint=0.0001,
                 kl_constraint=0.01,
                 alpha_mean_scale=1.0,
                 alpha_var_scale=100.0,
                 alpha_scale=10.0,
                 alpha_mean_max=0.1,
                 alpha_var_max=10.0,
                 alpha_max=1.0,
                 sample_action_num=64,
                 mstep_iteration_num=5,
                 actor_lr=0.003,
                 critic_lr=0.001,
                 ac_model="mlp",
                 hidden_sizes=[256, 256],
                 gamma=0.99,
                 polyak=0.995,
                 num_q=2,
                 **kwargs) -> None:
        r'''
        Constrained Variational Policy Optimization

        Args:
        @param env : The environment must satisfy the OpenAI Gym API.
        @param logger: Log useful informations, and help to save model
        :param dual_constraint:
        (float) hard constraint of the dual formulation in the E-step
        correspond to [2] p.4 ε
        @param kl_mean_constraint:
            (float) hard constraint of the mean in the M-step
            correspond to [2] p.6 ε_μ for continuous action space
        @param kl_var_constraint:
            (float) hard constraint of the covariance in the M-step
            correspond to [2] p.6 ε_Σ for continuous action space
        @param kl_constraint:
            (float) hard constraint in the M-step
            correspond to [2] p.6 ε_π for discrete action space
        @param discount_factor: (float) discount factor used in Policy Evaluation
        @param alpha_scale: (float) scaling factor of the lagrangian multiplier in the M-step, only used in Discrete action space
        @param sample_episode_num: the number of sampled episodes
        @param sample_episode_maxstep: maximum sample steps of an episode
        @param sample_action_num:
        @param batch_size: (int) size of the sampled mini-batch
        @param episode_rerun_num:
        @param mstep_iteration_num: (int) the number of iterations of the M-step
        @param evaluate_episode_maxstep: maximum evaluate steps of an episode
        @param actor_lr, critic_lr (float): Learning rate for policy and Q-value learning.
        @param ac_model: the actor critic model name
        @param gamma (float): Discount factor. (Always between 0 and 1.)
        @param polyak (float): Interpolation factor in polyak averaging for target 
        @param num_q (int): number of models in the q-ensemble critic.
        '''
        super().__init__()

        self.logger = logger
        self.dual_constraint = dual_constraint
        self.kl_mean_constraint = kl_mean_constraint
        self.kl_var_constraint = kl_var_constraint
        self.kl_constraint = kl_constraint
        self.alpha_mean_scale = alpha_mean_scale
        self.alpha_var_scale = alpha_var_scale
        self.alpha_scale = alpha_scale
        self.alpha_mean_max = alpha_mean_max
        self.alpha_var_max = alpha_var_max
        self.alpha_max = alpha_max
        self.sample_action_num = sample_action_num
        self.mstep_iteration_num = mstep_iteration_num
        self.gamma = gamma
        self.polyak = polyak
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.hidden_sizes = hidden_sizes
        self.use_cost_decay = use_cost_decay
        self.cost_start = cost_start
        self.cost_end = cost_end
        self.decay_epoch = decay_epoch

        ################ create actor critic model ###############
        self.obs_dim = env.observation_space.shape[0]
        self.act_dim = env.action_space.shape[0]
        # Action limit for normalization: critically, assumes all dimensions share the same bound!
        self.act_lim = env.action_space.high[0]

        if ac_model.lower() == "mlp":
            actor = CholeskyGaussianActor(self.obs_dim, self.act_dim, -self.act_lim,
                                          self.act_lim, hidden_sizes, nn.ReLU)
            critic = EnsembleQCritic(self.obs_dim,
                                     self.act_dim,
                                     hidden_sizes,
                                     nn.ReLU,
                                     num_q=num_q)
        else:
            raise ValueError(f"{ac_model} ac model does not support.")

        # Set up optimizer and target q models
        self._ac_training_setup(actor, critic)

        qc = EnsembleQCritic(self.obs_dim,
                             self.act_dim,
                             self.hidden_sizes,
                             nn.ReLU,
                             num_q=num_qc)
        self._qc_training_setup(qc)

        self.timeout_steps = timeout_steps

        if self.use_cost_decay:
            self.epoch = 0
            self.qc_start = self.cost_start * (1 - self.gamma**timeout_steps) / (
                1 - self.gamma) / timeout_steps
            self.qc_end = self.cost_end * (1 - self.gamma**timeout_steps) / (
                1 - self.gamma) / timeout_steps
            self.decay_func = lambda x: self.qc_end + (
                self.qc_start - self.qc_end) * np.exp(-5. * x / self.decay_epoch)
            self._step_qc_thres()

        else:
            self.qc_thres = cost_limit * (1 - self.gamma**timeout_steps) / (
                1 - self.gamma) / timeout_steps
        print("Cost constraint: ", self.qc_thres)

        self.eta = 0.1
        self.lam = 0.1
        self.alpha_mean = 0.0  # lagrangian multiplier for continuous action space in the M-step
        self.alpha_var = 0.0  # lagrangian multiplier for continuous action space in the M-step

        # Set up model saving
        self.save_model()

    def _step_qc_thres(self):
        self.qc_thres = self.decay_func(
            self.epoch) if self.epoch < self.decay_epoch else self.qc_end
        self.epoch += 1

    def _ac_training_setup(self, actor, critic):
        critic_targ = deepcopy(critic)
        actor_targ = deepcopy(actor)
        self.actor, self.actor_targ, self.critic, self.critic_targ = to_device(
            [actor, actor_targ, critic, critic_targ], get_device_name())
        # Freeze target networks with respect to optimizers (only update via polyak averaging)
        for p in self.critic_targ.parameters():
            p.requires_grad = False
        for p in self.actor_targ.parameters():
            p.requires_grad = False

        # Set up optimizers for policy and value function
        self.actor_optimizer = Adam(self.actor.parameters(), lr=self.actor_lr)
        self.critic_optimizer = Adam(self.critic.parameters(), lr=self.critic_lr)

    def _qc_training_setup(self, qc):
        qc_targ = deepcopy(qc)
        self.qc, self.qc_targ = to_device([qc, qc_targ], get_device_name())
        # Freeze target networks with respect to optimizers (only update via polyak averaging)
        for p in self.qc_targ.parameters():
            p.requires_grad = False

        # Set up optimizers for safety critic
        self.qc_optimizer = Adam(self.qc.parameters(), lr=self.critic_lr)

    def act(self, obs, deterministic=False, with_logprob=False):
        '''
        Given a single obs, return the action, logp.
        This API is used to interact with the env.

        @param obs (1d ndarray): observation
        @param deterministic (bool): True for evaluation mode, which returns the action with highest pdf (mean).
        @param with_logprob (bool): True to return log probability of the sampled action, False to return None
        @return act, logp, (1d ndarray)
        '''
        obs = to_tensor(obs).reshape(1, -1)
        logp_a = None
        with torch.no_grad():
            mean, cholesky, pi_dist = self.actor_forward(obs)
            a = mean if deterministic else pi_dist.sample()
            logp_a = pi_dist.log_prob(a) if with_logprob else None
        # squeeze them to the right shape
        a, logp_a = np.squeeze(to_ndarray(a), axis=0), np.squeeze(to_ndarray(logp_a))
        return a, logp_a

    def learn_on_batch(self, data: dict):
        '''
        Given a batch of data, train the policy
        data keys: (obs, act, rew, obs_next, done)
        '''
        self._update_critic(data)
        self._update_qc(data)
        # Freeze Q-networks so you don't waste computational effort
        # computing gradients for them during the policy learning step.
        # for p in self.critic.parameters():
        #     p.requires_grad = False
        # for p in self.qc.parameters():
        #     p.requires_grad = False

        self._update_actor(data)

        # Unfreeze Q-networks so you can optimize it at next DDPG step.
        # for p in self.critic.parameters():
        #     p.requires_grad = True
        # for p in self.qc.parameters():
        #     p.requires_grad = True

        # Finally, update target networks by polyak averaging.
        self._polyak_update_target(self.critic, self.critic_targ)
        self._polyak_update_target(self.qc, self.qc_targ)
        self._polyak_update_target(self.actor, self.actor_targ)

    def post_epoch_process(self):
        '''
        Update the cost limit.
        '''
        if self.use_cost_decay:
            self._step_qc_thres()

    def critic_forward(self, critic, obs, act):
        # return the minimum q values and the list of all q_values
        return critic.predict(obs, act)

    def actor_forward(self, obs, return_pi=True):
        r''' 
        Return action distribution and action log prob [optional].
        @param obs, (tensor), [batch, obs_dim]
        @return mean, (tensor), [batch, act_dim]
        @return cholesky, (tensor), (batch, act_dim, act_dim)
        @return pi_dist, (MultivariateNormal)
        '''
        mean, cholesky = self.actor(obs)
        pi_dist = MultivariateNormal(mean, scale_tril=cholesky) if return_pi else None
        return mean, cholesky, pi_dist

    def _update_actor(self, data):
        '''
        Update the actor network
        '''
        obs = data['obs']  # [batch, obs_dim]
        N = self.sample_action_num
        K = obs.shape[0]
        da = self.act_dim
        ds = self.obs_dim

        with torch.no_grad():
            # sample N actions per state
            b_mean, b_A = self.actor_targ.forward(obs)  # (K,)
            b = MultivariateNormal(b_mean, scale_tril=b_A)  # (K,)
            sampled_actions = b.sample((N, ))  # (N, K, da)

            expanded_states = obs[None, ...].expand(N, -1, -1)  # (N, K, ds)
            target_q, _ = self.critic_forward(self.critic_targ,
                                              expanded_states.reshape(-1, ds),
                                              sampled_actions.reshape(-1, da))
            target_q = target_q.reshape(N, K)  # (N, K)
            target_q_np = to_ndarray(target_q).T  # (K, N)
            target_qc, _ = self.critic_forward(self.qc_targ,
                                               expanded_states.reshape(-1, ds),
                                               sampled_actions.reshape(-1, da))
            target_qc = target_qc.reshape(N, K)  # (N, K)
            target_qc_np = to_ndarray(target_qc).T  # (K, N)

        def dual(x):
            """
            dual function of the non-parametric variational
            """
            η, lam = x
            target_q_np_comb = target_q_np - lam * target_qc_np
            max_q = np.max(target_q_np_comb, 1)
            return η * self.dual_constraint + lam * self.qc_thres + np.mean(max_q) \
                + η * np.mean(np.log(np.mean(np.exp((target_q_np_comb - max_q[:, None]) / η), axis=1)))

        bounds = [(1e-6, 1e5), (1e-6, 1e5)]
        options = {"ftol": 1e-3, "maxiter": 10}
        res = minimize(dual,
                       np.array([self.eta, self.lam]),
                       method='SLSQP',
                       bounds=bounds,
                       tol=1e-3,
                       options=options)
        self.eta, self.lam = res.x

        qij = torch.softmax((target_q - self.lam * target_qc) / self.eta,
                            dim=0)  # (N, K) or (da, K)

        # M-Step of Policy Improvement
        # [2] 4.2 Fitting an improved policy (Step 3)
        for _ in range(self.mstep_iteration_num):
            mean, A = self.actor.forward(obs)
            # First term of last eq of [2] p.5
            # see also [2] 4.2.1 Fitting an improved Gaussian policy
            π1 = MultivariateNormal(loc=mean, scale_tril=b_A)  # (K,)
            π2 = MultivariateNormal(loc=b_mean, scale_tril=A)  # (K,)
            loss_p = torch.mean(qij * (
                π1.expand((N, K)).log_prob(sampled_actions)  # (N, K)
                + π2.expand((N, K)).log_prob(sampled_actions)  # (N, K)
            ))

            kl_μ, kl_Σ, Σi_det, Σ_det = gaussian_kl(μi=b_mean, μ=mean, Ai=b_A, A=A)

            if np.isnan(kl_μ.item()):  # This should not happen
                raise RuntimeError('kl_μ is nan')
            if np.isnan(kl_Σ.item()):  # This should not happen
                raise RuntimeError('kl_Σ is nan')

            # Update lagrange multipliers by gradient descent
            # this equation is derived from last eq of [2] p.5,
            # just differentiate with respect to α
            # and update α so that the equation is to be minimized.
            self.alpha_mean -= self.alpha_mean_scale * (self.kl_mean_constraint -
                                                        kl_μ).detach().item()
            self.alpha_var -= self.alpha_var_scale * (self.kl_var_constraint -
                                                      kl_Σ).detach().item()

            self.alpha_mean = np.clip(self.alpha_mean, 0.0, self.alpha_mean_max)
            self.alpha_var = np.clip(self.alpha_var, 0.0, self.alpha_var_max)

            self.actor_optimizer.zero_grad()
            # last eq of [2] p.5
            loss_l = -(loss_p + self.alpha_mean *
                       (self.kl_mean_constraint - kl_μ) + self.alpha_var *
                       (self.kl_var_constraint - kl_Σ))

            loss_l.backward()
            clip_grad_norm_(self.actor.parameters(), 0.01)
            self.actor_optimizer.step()

            # Log actor update info
            self.logger.store(LossAll=loss_l.item(),
                              LossMLE=(-loss_p).item(),
                              mean_Σ_det=Σ_det.item(),
                              max_kl_Σ=kl_Σ.item(),
                              max_kl_μ=kl_μ.item(),
                              QcThres=self.qc_thres,
                              QcValue=target_qc_np,
                              eta=self.eta,
                              lam=self.lam)

    def _update_critic(self, data):
        '''
        Update the critic network
        '''
        def critic_loss():
            obs, act, reward, obs_next, done = to_tensor(data['obs']), to_tensor(
                data['act']), to_tensor(data['rew']), to_tensor(
                    data['obs2']), to_tensor(data['done'])

            _, q_list = self.critic_forward(self.critic, obs, act)
            # Bellman backup for Q functions
            with torch.no_grad():
                # Target actions come from *current* policy
                _, logp_a_next, pi_dist = self.actor_forward(obs_next)
                act_next = pi_dist.sample()
                # Target Q-values
                q_pi_targ, _ = self.critic_forward(self.critic_targ, obs_next, act_next)
                backup = reward + self.gamma * (1 - done) * q_pi_targ
            # MSE loss against Bellman backup
            loss_q = self.critic.loss(backup, q_list)
            # Useful info for logging
            q_info = dict()
            for i, q in enumerate(q_list):
                q_info["QVals" + str(i)] = to_ndarray(q)
            return loss_q, q_info

        # First run one gradient descent step for Q1 and Q2
        self.critic_optimizer.zero_grad()
        loss_critic, loss_q_info = critic_loss()
        loss_critic.backward()
        self.critic_optimizer.step()

        # Log critic update info
        # Record things
        self.logger.store(LossQ=loss_critic.item(), **loss_q_info)

    def _update_qc(self, data):
        '''
        Update the qc network
        '''
        def critic_loss():
            obs, act, cost, obs_next, done = to_tensor(data['obs']), to_tensor(
                data['act']), to_tensor(data['cost']), to_tensor(
                    data['obs2']), to_tensor(data['done'])

            _, q_list = self.critic_forward(self.qc, obs, act)
            # Bellman backup for Q functions
            with torch.no_grad():
                # Target actions come from *current* policy
                _, logp_a_next, pi_dist = self.actor_forward(obs_next)
                act_next = pi_dist.sample()
                # Target Q-values
                q_pi_targ, _ = self.critic_forward(self.qc_targ, obs_next, act_next)
                # backup = cost + self.gamma * (1 - done) * q_pi_targ
                backup = cost + self.gamma * q_pi_targ
            # MSE loss against Bellman backup
            loss_q = self.qc.loss(backup, q_list)
            # Useful info for logging
            q_info = dict()
            for i, q in enumerate(q_list):
                q_info["QCVals" + str(i)] = to_ndarray(q)
            return loss_q, q_info

        # First run one gradient descent step for Q1 and Q2
        self.qc_optimizer.zero_grad()
        loss_qc, loss_qc_info = critic_loss()
        loss_qc.backward()
        self.qc_optimizer.step()

        # Log critic update info
        # Record things
        self.logger.store(LossQC=loss_qc.item(), **loss_qc_info)

    def _polyak_update_target(self, model, model_targ):
        '''
        Update target networks by polyak averaging.
        '''
        with torch.no_grad():
            for p, p_targ in zip(model.parameters(), model_targ.parameters()):
                # NB: We use an in-place operations "mul_", "add_" to update target
                # params, as opposed to "mul" and "add", which would make new tensors.
                p_targ.data.mul_(self.polyak)
                p_targ.data.add_((1 - self.polyak) * p.data)

    def save_model(self):
        actor, critic, qc = self.actor, self.critic, self.qc
        self.logger.setup_pytorch_saver((actor, critic, qc))

    def load_model(self, path):
        actor, critic, qc = torch.load(path)
        actor, critic, qc = to_device([actor, critic, qc])
        self._ac_training_setup(actor, critic)
        self._qc_training_setup(qc)
