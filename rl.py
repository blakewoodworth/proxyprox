import torch
import gymnasium as gym
import numpy as np
import argparse
from tqdm import tqdm

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

def rtg(rewards, g):
    rtgs = [rewards[-1]]
    for rew in reversed(rewards):
        rtgs.append(rew + g*rtgs[-1])
    rtgs.reverse()
    return torch.Tensor(rtgs)

def generalized_advantage_estimate(rewards, values, g, l):
    deltas = np.array(rewards) - np.array(values)
    deltas[:-1] += g*np.array(values)[1:]

    backward_gaes = [deltas[-1]]
    for i in reversed(range(len(rewards)-1)):
        backward_gaes.append(deltas[i] + g*l*backward_gaes[-1])
    backward_gaes.reverse()
    backward_gaes = torch.Tensor(backward_gaes)
    return (backward_gaes - backward_gaes.mean())/backward_gaes.std()


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
            obs, _ = env.env.reset()
            while not done:
                episode_t += 1

                observations.append(torch.Tensor(obs))

                with torch.no_grad():
                    action_idx, logprob, _, value = agent.action_and_value(observations[-1])
                actions.append(action_idx)
                logprobs.append(logprob)
                episode_values.append(value.item())
                action = env.action_map[action_idx]

                obs, reward, terminated, truncated, _ = env.env.step(action)

                total_episode_reward += reward
                episode_rewards.append(reward)

                done = (episode_t > args['max_time_steps_per_episode']) or terminated or truncated
            
            total_reward += total_episode_reward
            reward_to_gos.append(rtg(episode_rewards, args['gamma']))
            advantages.append(generalized_advantage_estimate(episode_rewards, episode_values, args['gamma'], args['gae_lambda']))

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


def ppo_clip(env, agent, args):
    optimizers = {}
    schedulers = {}
    target_grads = {}
    verb = 'deprecated' if int(torch.__version__[0])>1 else False
    for name, parameters in agent.named_parameters():
        optimizers[name] = torch.optim.Adam([parameters], lr=args['stepsize'])
        schedulers[name] = torch.optim.lr_scheduler.CosineAnnealingLR(optimizers[name], 
            args['n_updates'], eta_min=args['stepsize_anneal_factor']*args['stepsize'], verbose=verb)
        target_grads[name] = torch.zeros(parameters.size())
    parameter_names = list(optimizers.keys())
    
    avg_rewards = []
    for i_update in range(1,args['n_updates']+1):

        # trajectory = collect_trajectories(env, agent, args)
        trajectory = Trajectory(env, agent, args)
        avg_rewards.append(trajectory.avg_reward)

        if i_update % 50 == 0:
            print(f"Update {i_update}/{args['n_updates']}: average reward {np.mean(avg_rewards):.3f}")
            avg_rewards = []

        # optimize PPO-clip objective, value function, and entropy loss
        for inner_epoch in range(args['inner_epochs']):
            
            for i in range(trajectory.steps_per_epoch):

                batch_obs, batch_actions, batch_rtg, batch_logprob, batch_advantage = trajectory.get_batch()

                # calculate ppo-clip loss
                _, new_logprob, new_entropy, new_value = agent.action_and_value(batch_obs, batch_actions)
                ratio = (new_logprob - batch_logprob).exp()

                ppo_clip_loss1 = -ratio * batch_advantage
                ppo_clip_loss2 = -torch.clamp(ratio, 1-args['clip_epsilon'], 1+args['clip_epsilon'])
                ppo_clip_loss = torch.max(ppo_clip_loss1, ppo_clip_loss2).mean()

                # calculate value loss
                # value_loss = ((new_value - batch_rtg)**2).mean()
                value_loss = torch.nn.functional.smooth_l1_loss(new_value.view(-1), batch_rtg)

                # calculate entropy loss
                entropy_loss = -new_entropy.mean()

                all_loss = ppo_clip_loss + args['value_coefficient']*value_loss + args['entropy_coefficient']*entropy_loss

                for name in parameter_names:
                    optimizers[name].zero_grad()
                    target_grads[name] = torch.zeros(target_grads[name].size())

                all_loss.backward()

                # for name, parameters in agent.named_parameters():
                #     print(name, parameters.grad)
                #     print('\n\n')

                # explode()

                for name in parameter_names:
                    optimizers[name].step()
        for name in parameter_names:
            schedulers[name].step()

class DoublePendulum:

    def __init__(self, n_actions=25):
        self.env = gym.make('InvertedDoublePendulum-v5')
        self.n_actions = n_actions
        self.action_map = np.linspace(-1.,1.,self.n_actions)[:,None]


class SinglePendulum:

    def __init__(self, n_actions=25):
        self.env = gym.make('InvertedPendulum-v5')
        self.n_actions = n_actions
        self.action_map = np.linspace(-3.,3.,self.n_actions)[:,None]



np.random.seed(1)
torch.manual_seed(1)

args = {
    'n_actions': 21,
    'hidden_dim': 128,
    'time_steps_per_batch': 800,
    'max_time_steps_per_episode': 200,
    'n_updates': 1000,
    'inner_epochs': 4,
    'batch_size': 250,
    'clip_epsilon': 0.2,
    'value_coefficient': 0.01,
    'entropy_coefficient': 0.005,
    'gamma': 0.975,
    'gae_lambda': 0.975,
    'stepsize': 5e-4,
    'stepsize_anneal_factor': 1e-1
}

target_env = DoublePendulum(n_actions=args['n_actions'])
# proxy_env = SinglePendulum(n_actions=target_env.n_actions)

args['observation_shape'] = target_env.env.observation_space.shape
args['action_n'] = target_env.n_actions

agent = Agent(args)

ppo_clip(target_env, agent, args)

# env = gym.make(args['env_name'], render_mode='rgb_array')
# env = gym.wrappers.RecordVideo(env, f"videos/{args['env_name']}")

# obs, _ = env.reset()
# done = False
# rewards = 0.
# while not done:
#     with torch.no_grad():
#         action = agent.action_and_value(torch.Tensor(obs))[0]
#     obs, reward, terminated, truncated, _ = env.step(action.numpy())
#     rewards += reward
#     done = terminated or truncated
# print(f'Final simulation total reward: {rewards}')
# env.close()



