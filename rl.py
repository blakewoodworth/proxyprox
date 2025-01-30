import torch
import gymnasium as gym
import numpy as np
import argparse
from tqdm import tqdm
from matplotlib import pyplot as plt


class DoublePendulum:

    def __init__(self, n_actions=25):
        self.env = gym.make('InvertedDoublePendulum-v5')
        self.n_actions = n_actions
        self.action_map = np.linspace(-1.,1.,self.n_actions)[:,None]

    def reset(self):
        return self.env.reset()

    def step(self, action):
        return self.env.step(action)


class SinglePendulum:

    def __init__(self, n_actions=25):
        self.env = gym.make('InvertedPendulum-v5')
        self.n_actions = n_actions
        self.action_map = np.linspace(-3.,3.,self.n_actions)[:,None]

    def obs_mapping(self, obs):
        return [obs[0], np.sin(obs[1]), 0., np.cos(obs[1]), 0., obs[2], obs[3], 0., 0.]

    def reset(self):
        obs, x = self.env.reset()
        return self.obs_mapping(obs), x

    def step(self, action):
        obs, reward, terminated, truncated, x = self.env.step(action)
        return self.obs_mapping(obs), reward, terminated, truncated, x
        

class Agent(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.actor = torch.nn.Sequential(
            torch.nn.Linear(np.array(args['observation_shape']).prod(), args['hidden_dim']),
            torch.nn.GELU(),
            torch.nn.Linear(args['hidden_dim'], args['hidden_dim']),
            torch.nn.GELU(),
            torch.nn.Linear(args['hidden_dim'], args['action_n'])
        )
        self.critic = torch.nn.Sequential(
            torch.nn.Linear(np.array(args['observation_shape']).prod(), args['hidden_dim']),
            torch.nn.GELU(),
            torch.nn.Linear(args['hidden_dim'], args['hidden_dim']),
            torch.nn.GELU(),
            torch.nn.Linear(args['hidden_dim'], 1)
        )

    def action_and_value(self, obs, action=None):
        probs = torch.distributions.Categorical(logits=self.actor(obs))
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(obs)


class Trajectory:

    def __init__(self, env, agent, args):
        observations = []
        actions = []
        reward_to_gos = []
        logprobs = []
        advantages = []

        total_reward = 0.
        n_episodes = 0.
        
        t = 0
        while t < args['time_steps_per_batch']:
            n_episodes += 1
            total_episode_reward = 0.
            episode_t = 0
            done = False
            episode_rewards = []
            episode_values = []
            obs, _ = env.reset()
            while not done:
                episode_t += 1

                observations.append(torch.Tensor(obs))

                with torch.no_grad():
                    action_idx, logprob, _, value = agent.action_and_value(observations[-1])
                actions.append(action_idx)
                logprobs.append(logprob)
                episode_values.append(value.item())
                action = env.action_map[action_idx]

                obs, reward, terminated, truncated, _ = env.step(action)

                total_episode_reward += reward
                episode_rewards.append(reward)

                done = (episode_t > args['max_time_steps_per_episode']) or terminated or truncated
            
            total_reward += total_episode_reward
            reward_to_gos.append(self.rtg(episode_rewards, args['gamma']))
            advantages.append(self.generalized_advantage_estimate(episode_rewards, episode_values, args['gamma'], args['gae_lambda']))

            t += episode_t

        self.observations = torch.stack(observations)
        self.actions = torch.Tensor(actions)
        self.reward_to_gos = torch.Tensor(torch.cat(reward_to_gos))
        self.logprobs = torch.Tensor(logprobs)
        self.advantages = torch.Tensor(torch.cat(advantages))
        self.avg_reward = total_reward/n_episodes

        self.batch_size = args['batch_size']
        self.n_times = self.observations.shape[0]
        self.steps_per_epoch = self.n_times // self.batch_size

        self.reset_batch()

    def rtg(self, rewards, g):
        rtgs = [rewards[-1]]
        for rew in reversed(rewards):
            rtgs.append(rew + g*rtgs[-1])
        rtgs.reverse()
        return torch.Tensor(rtgs)

    def generalized_advantage_estimate(self, rewards, values, g, l):
        deltas = np.array(rewards) - np.array(values)
        deltas[:-1] += g*np.array(values)[1:]

        backward_gaes = [deltas[-1]]
        for i in reversed(range(len(rewards)-1)):
            backward_gaes.append(deltas[i] + g*l*backward_gaes[-1])
        backward_gaes.reverse()
        backward_gaes = torch.Tensor(backward_gaes)
        return (backward_gaes - backward_gaes.mean())/backward_gaes.std()

    def reset_batch(self):
        self.batch_idx = 0
        self.batch_order = np.random.permutation(self.n_times-(self.n_times%self.batch_size))

    def get_batch(self):
        if self.batch_idx >= len(self.batch_order):
            self.reset_batch()
        batch_idxs = self.batch_order[self.batch_idx:self.batch_idx+self.batch_size]
        self.batch_idx += self.batch_size

        batch_obs = self.observations[batch_idxs]
        batch_actions = self.actions[batch_idxs]
        batch_rtg = self.reward_to_gos[batch_idxs]
        batch_logprob = self.logprobs[batch_idxs]
        batch_advantage = self.advantages[batch_idxs]

        return batch_obs, batch_actions, batch_rtg, batch_logprob, batch_advantage


class ProxyProxRL:

    def __init__(self, agent, target_env, proxy_env, args):
        self.agent = agent
        self.target_env = target_env
        self.proxy_env = proxy_env
        self.args = args

    def calculate_rl_loss(self, batch_obs, batch_actions, batch_rtg, batch_logprob, batch_advantage):
        # calculate ppo-clip loss
        _, new_logprob, new_entropy, new_value = self.agent.action_and_value(batch_obs, batch_actions)
        ratio = (new_logprob - batch_logprob).exp()

        ppo_clip_loss1 = -ratio * batch_advantage
        ppo_clip_loss2 = -torch.clamp(ratio, 1-self.args['clip_epsilon'], 1+self.args['clip_epsilon'])
        ppo_clip_loss = torch.max(ppo_clip_loss1, ppo_clip_loss2).mean()

        # calculate value loss
        value_loss = torch.nn.functional.smooth_l1_loss(new_value.view(-1), batch_rtg)

        # calculate entropy loss
        entropy_loss = -new_entropy.mean()

        return ppo_clip_loss + self.args['value_coefficient']*value_loss + self.args['entropy_coefficient']*entropy_loss

    def calculate_grad(self, trajectory):
        grads = {}
        for name, parameters in self.agent.named_parameters():
            grads[name] = torch.zeros(parameters.size())

        for i in range(trajectory.steps_per_epoch):
            loss = self.calculate_rl_loss(*trajectory.get_batch())
            self.agent.zero_grad()
            loss.backward()

            for name, parameters in self.agent.named_parameters():
                grads[name] += parameters.grad.detach()

        return grads

    def get_bias_correction(self):
        
        target_trajectory = Trajectory(self.target_env, self.agent, self.args)
        target_grads = self.calculate_grad(target_trajectory)
        proxy_grads = self.calculate_grad(Trajectory(self.proxy_env, self.agent, self.args))

        bias_correction = {}
        for name in target_grads:
            bias_correction[name] = target_grads[name] - proxy_grads[name]

        return bias_correction, target_trajectory.avg_reward.item()

    def train(self, do_sgd=False):
        self.optimizers = {}
        # self.schedulers = {}
        for name, parameters in self.agent.named_parameters():
            self.optimizers[name] = torch.optim.Adam([parameters], lr=self.args['stepsize'])
            # self.schedulers[name] = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizers[name], self.args['n_outer_loops'], 
            #                                 eta_min=self.args['stepsize_anneal_factor']*self.args['stepsize'], verbose='deprecated')
        self.parameter_names = list(self.optimizers.keys())
        
        avg_rewards = []
        n_outer = self.args['n_outer_loops']
        for k in range(1,n_outer+1):

            if do_sgd:
                target_trajectory = Trajectory(self.target_env, self.agent, self.args)
                avg_rewards.append(target_trajectory.avg_reward.item())

                for _ in range(target_trajectory.steps_per_epoch):

                    loss = self.calculate_rl_loss(*target_trajectory.get_batch())

                    if loss.item() > 1000:
                        return None

                    for optimizer in self.optimizers.values():
                        optimizer.zero_grad()

                    loss.backward()

                    for optimizer in self.optimizers.values():
                        optimizer.step() 
            else:
                bias_correction, target_avg_reward = self.get_bias_correction()
                avg_rewards.append(target_avg_reward)
                wk = {n: p.detach().clone() for n, p in self.agent.named_parameters()}
                
                for i in range(self.args['n_inner_loops']):

                    proxy_trajectory = Trajectory(self.proxy_env, self.agent, self.args)
                    for _ in range(self.args['n_inner_inner_epochs']):
                        for _ in range(proxy_trajectory.steps_per_epoch):

                            loss = self.calculate_rl_loss(*proxy_trajectory.get_batch())

                            if loss.item() > 1000:
                                return None

                            for optimizer in self.optimizers.values():
                                optimizer.zero_grad()

                            loss.backward()

                            for name, parameters in self.agent.named_parameters():
                                parameters.grad += (parameters - wk[name]) / self.args['eta']
                                parameters.grad += bias_correction[name]

                            for optimizer in self.optimizers.values():
                                optimizer.step() 

            # for scheduler in self.schedulers.values():
            #     scheduler.step()

        return avg_rewards





np.random.seed(1)

n_actions = 21

target_env = DoublePendulum(n_actions=n_actions)
proxy_env = SinglePendulum(n_actions=n_actions)

args = {
    'n_actions': n_actions,
    'hidden_dim': 25,
    'time_steps_per_batch': 400,
    'max_time_steps_per_episode': 200,
    'n_outer_loops': 50,
    'n_inner_loops': 4,
    'n_inner_inner_epochs': 1,
    'batch_size': 200,
    'clip_epsilon': 0.15,
    'value_coefficient': 0.03,
    'entropy_coefficient': 0.001,
    'gamma': 0.99,
    'gae_lambda': 0.99,
    'stepsize': 7e-3,
    'stepsize_anneal_factor': 1e-1,
    'eta': 1.,
    'observation_shape': target_env.env.observation_space.shape,
    'action_n': target_env.n_actions
}

n_reps = 4

lrs = np.power(10, np.linspace(-2.1, -1.5, 5))
best_reward = -float('inf')
best_avg_rewards = None
best_lr = None
progress_bar = tqdm(total=len(lrs)*n_reps)
for lr in lrs:
    args['stepsize'] = lr

    avg_avg_rewards = None
    reward = 0.
    n_converged = 0
    for _ in range(n_reps):
        torch.manual_seed(1)
        agent = Agent(args)
        avg_rewards = ProxyProxRL(agent, target_env, proxy_env, args).train(do_sgd=True)
        if avg_rewards is not None:
            n_converged += 1
            reward += np.mean(np.array(avg_rewards)[-10:])
            if avg_avg_rewards is None:
                avg_avg_rewards = np.array(avg_rewards)
            else:
                avg_avg_rewards += np.array(avg_rewards)
        progress_bar.update()
    if n_converged > 0:
        reward /= n_converged
        avg_avg_rewards /= n_converged
    else:
        reward = -float('inf')
    if reward > best_reward:
        best_reward = reward
        best_avg_rewards = avg_avg_rewards
        best_lr = lr

print(f'\n\nbest lr: {best_lr} | best sgd reward: {best_reward}')

sgd_avg_rewards = best_avg_rewards


info = []
# lrs = np.power(10, np.linspace(-3, -1.5, 10))
# etas = np.power(10, np.linspace(-1., 1., 10))

pairs = [
    (0.002, 5),
    (0.002, 10),
    (0.002, 20),
    (0.002, 40),
    (0.004, 1),
    (0.004, 4),
    (0.004, 8),
    (0.004, 12),
    (0.008, 1),
    (0.008, 4),
    (0.008, 8),
    (0.008, 12),
]
progress_bar = tqdm(total=len(pairs)*n_reps)

for lr, eta in pairs:
    args['stepsize'] = lr
    args['eta'] = eta

    avg_avg_rewards = None
    reward = 0.
    n_converged = 0
    for _ in range(n_reps):
        torch.manual_seed(1)
        agent = Agent(args)
        avg_rewards = ProxyProxRL(agent, target_env, proxy_env, args).train()
        if avg_rewards is not None:
            n_converged += 1
            reward += np.mean(np.array(avg_rewards)[-10:])
            if avg_avg_rewards is None:
                avg_avg_rewards = np.array(avg_rewards)
            else:
                avg_avg_rewards += np.array(avg_rewards)
        progress_bar.update()
    if n_converged > 0:
        info.append((lr, eta, reward/n_converged, avg_avg_rewards/n_converged))
        print(f'lr: {lr:.4f}, eta: {eta:.2f}, average reward: {reward/n_converged:.2f}')

n_best = 5
info.sort(key=lambda x: x[2], reverse=True)

print('\n\n')
for i in range(n_best):
    lr, eta, end, avg_rewards = info[i]
    print(f'{i}th best | lr: {lr} | eta: {eta} | end reward: {end}')

plt.figure()
plt.title('Target: Double Pendulum, Proxy: Single Pendulum')
plt.plot(sgd_avg_rewards, label='SGD baseline')
for i in range(n_best):
    lr, eta, _, avg_rewards = info[i]
    plt.plot(avg_rewards, label=f'lr={lr}, eta={eta}')
plt.legend()
plt.savefig('plots/pendulum-rl.pdf')
plt.show()













