import torch
import torch.nn.functional as F
from stable_baselines3.common import utils as sb3_utils
from torch import optim
from torch.distributions import Categorical

# Sources:
# overview https://www.endtoend.ai/blog/pytorch-pg-implementations/
# https://github.com/higgsfield/RL-Adventure-2/blob/master/1.actor-critic.ipynb
# https://github.com/pytorch/examples/blob/master/reinforcement_learning/actor_critic.py
from algs.reinforce import network, reinforce_helper
from utils import helper


def run(params):
    device, use_cuda = helper.get_pytorch_device()
    sb3_utils.set_random_seed(params.seed, using_cuda=use_cuda)
    env = helper.make_env(params)

    model = network.get_model_class(params)(env).to(device)
    optimizer = optim.Adam(model.parameters(), lr=params.learning_rate)

    ep_no = 0
    total_rewards = []

    while ep_no < params.num_episodes:
        ep_no += 1
        state_vals, log_probs = [], []

        # unroll the policy
        state = env.reset()
        rewards = []
        # for each episode, only run 10_000 steps so that we don't
        # infinite loop while learning
        for t in range(10_000):
            state = torch.from_numpy(state).float().unsqueeze(0).to(device)
            probs, state_value = model(state)
            c = Categorical(probs)
            action = c.sample()
            log_probs.append(c.log_prob(action))
            state_vals.append(state_value)

            state, reward, done, info = env.step(action.item())

            rewards.append(reward)
            if done:
                break
        total_rewards.append(sum(rewards))

        reinforce_helper.log_results(ep_no, total_rewards, info, t, params)

        returns = reinforce_helper.discount_rewards(rewards, params)


        # backprop
        state_vals = torch.stack(state_vals).squeeze()
        returns = returns.to(device)
        policy_loss = (-torch.stack(log_probs).squeeze(1) * (returns - state_vals).detach()).mean()
        baseline_loss = F.smooth_l1_loss(state_vals, returns, reduction='mean')

        # reset gradients
        optimizer.zero_grad()
        loss = policy_loss + params.scaling_factor * baseline_loss
        loss.backward()
        optimizer.step()


if __name__ == '__main__':
    parser = reinforce_helper.parse_args()
    params = helper.get_parsed_params(parser)
    run(params)