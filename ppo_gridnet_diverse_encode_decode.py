# http://proceedings.mlr.press/v97/han19a/han19a.pdf

import os
import random
import time

import numpy as np
import torch
import torch.optim as optim
from gym.spaces import MultiDiscrete
from gym_microrts import microrts_ai
from gym_microrts.envs.vec_env import MicroRTSGridModeVecEnv
from stable_baselines3.common.vec_env import VecVideoRecorder
from torch import Tensor, nn
from torch.utils.tensorboard import SummaryWriter

from ppo_gridnet_diverse_encode_decode.argparser import (
    PPOGridNetDiverseEncodeDecodeArgumentParser,
)
from ppo_gridnet_diverse_encode_decode.stats_recorder import MicroRTSStatsRecorder
from ppo_gridnet_diverse_encode_decode.vec_monitor import VecMonitor
from ppo_gridnet_diverse_encode_decode.agent import Agent

parser = PPOGridNetDiverseEncodeDecodeArgumentParser()

args = parser.parse_args()
if not args.seed:
    args.seed = int(time.time())
args.num_envs = args.num_selfplay_envs + args.num_bot_envs
args.batch_size = int(args.num_envs * args.num_steps)
args.minibatch_size = int(args.batch_size // args.n_minibatch)


# TRY NOT TO MODIFY: setup the environment
experiment_name = f"{args.gym_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
writer = SummaryWriter(f"runs/{experiment_name}")
writer.add_text(
    "hyperparameters",
    "|param|value|\n|-|-|\n%s"
    % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
)
if args.prod_mode:
    import wandb

    run = wandb.init(
        project=args.wandb_project_name,
        entity=args.wandb_entity,
        # sync_tensorboard=True,
        config=vars(args),
        name=experiment_name,
        monitor_gym=True,
        save_code=True,
    )
    wandb.tensorboard.patch(save=False)
    writer = SummaryWriter(f"/tmp/{experiment_name}")
    CHECKPOINT_FREQUENCY = 50

# TRY NOT TO MODIFY: seeding
device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.backends.cudnn.deterministic = args.torch_deterministic
envs = MicroRTSGridModeVecEnv(
    num_selfplay_envs=args.num_selfplay_envs,
    num_bot_envs=args.num_bot_envs,
    max_steps=2000,
    render_theme=2,
    ai2s=[microrts_ai.coacAI for _ in range(args.num_bot_envs - 6)]
    + [microrts_ai.randomBiasedAI for _ in range(min(args.num_bot_envs, 2))]
    + [microrts_ai.lightRushAI for _ in range(min(args.num_bot_envs, 2))]
    + [microrts_ai.workerRushAI for _ in range(min(args.num_bot_envs, 2))],
    map_path="maps/16x16/basesWorkers16x16.xml",
    reward_weight=np.array([10.0, 1.0, 1.0, 0.2, 1.0, 4.0]),
)
envs = MicroRTSStatsRecorder(envs, args.gamma)
envs = VecMonitor(envs)
if args.capture_video:
    envs = VecVideoRecorder(
        envs,
        f"videos/{experiment_name}",
        record_video_trigger=lambda x: x % 1000000 == 0,
        video_length=2000,
    )
# if args.prod_mode:
#     envs = VecPyTorch(
#         SubprocVecEnv([make_env(args.gym_id, args.seed+i, i) for i in range(args.num_envs)], "fork"),
#         device
#     )
assert isinstance(
    envs.action_space, MultiDiscrete
), "only MultiDiscrete action space is supported"


agent = Agent(envs=envs, device=device).to(device)
optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)
if args.anneal_lr:
    # https://github.com/openai/baselines/blob/ea25b9e8b234e6ee1bca43083f8f3cf974143998/baselines/ppo2/defaults.py#L20
    lr = lambda f: f * args.learning_rate

# ALGO Logic: Storage for epoch data
mapsize = 16 * 16

# envs.action_space = MultiDiscrete([256   6   4   4   4   4   7  49])

# of shape (256, 7)
action_space_shape = (mapsize, envs.action_space.shape[0] - 1)

# of shape (256, 79)
invalid_action_shape = (mapsize, envs.action_space.nvec[1:].sum() + 1)

# envs.observation_space.shape = (h, w, n_features) = (16, 16, 27)

# of shape (num_steps, num_envs, h, w, n_features) = (256, 24, 16, 16, 27)
obs = torch.zeros((args.num_steps, args.num_envs) + envs.observation_space.shape).to(
    device
)

# of shape (num_steps, num_envs, h*w, num_discrete_actions) = (256, 24, 256, 7)
actions: Tensor = torch.zeros((args.num_steps, args.num_envs) + action_space_shape).to(
    device
)

# of shape (num_steps, num_envs) = (256, 24)
logprobs: Tensor = torch.zeros((args.num_steps, args.num_envs)).to(device)
rewards: Tensor = torch.zeros((args.num_steps, args.num_envs)).to(device)
dones: Tensor = torch.zeros((args.num_steps, args.num_envs)).to(device)
values: Tensor = torch.zeros((args.num_steps, args.num_envs)).to(device)

# of shape (num_steps, num_envs, mapsize, 79) = (256, 24, 256, 79)
invalid_action_masks: Tensor = torch.zeros(
    (args.num_steps, args.num_envs) + invalid_action_shape
).to(device)

# TRY NOT TO MODIFY: start the game
global_step = 0
start_time = time.time()

# Note how `next_obs` and `next_done` are used; their usage is equivalent to
# https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail/blob/84a7582477fb0d5c82ad6d850fe476829dddd2e1/a2c_ppo_acktr/storage.py#L60

# of shape (num_envs, h, w, n_features) = (24, 16, 16, 27)
next_obs = torch.Tensor(envs.reset()).to(device)

# of shape (num_envs) = (24)
next_done = torch.zeros(args.num_envs).to(device)

num_updates = args.total_timesteps // args.batch_size  # 16276

## CRASH AND RESUME LOGIC:
starting_update = 1
from jpype.types import JArray, JInt

if args.prod_mode and wandb.run.resumed:
    starting_update = run.summary.get("charts/update") + 1
    global_step = starting_update * args.batch_size
    api = wandb.Api()
    run = api.run(f"{run.entity}/{run.project}/{run.id}")
    model = run.file("agent.pt")
    model.download(f"models/{experiment_name}/")
    agent.load_state_dict(
        torch.load(f"models/{experiment_name}/agent.pt", map_location=device)
    )
    agent.eval()
    print(f"resumed at update {starting_update}")

print("Model's state_dict:")
for param_tensor in agent.state_dict():
    print(param_tensor, "\t", agent.state_dict()[param_tensor].size())
total_params = sum([param.nelement() for param in agent.parameters()])
print("Model's total parameters:", total_params)

for update in range(starting_update, num_updates + 1):
    # Annealing the rate if instructed to do so.
    if args.anneal_lr:
        frac = 1.0 - (update - 1.0) / num_updates
        lrnow = lr(frac)
        optimizer.param_groups[0]["lr"] = lrnow

    # TRY NOT TO MODIFY: prepare the execution of the game.
    for step in range(0, args.num_steps):
        # envs.render()
        global_step += 1 * args.num_envs
        obs[step] = next_obs
        dones[step] = next_done
        # ALGO LOGIC: put action logic here
        with torch.no_grad():
            values[step] = agent.get_value(obs[step]).flatten()
            # action of shape (num_envs, h*w, num_predicted_parameters) = (24, 256, 7)
            # logproba of shape (num_envs) = (24)
            # entropy of shape (num_envs) = (24)
            # invalid_action_masks of shape (num_envs, h*w, 79) = (24, 256, 79)
            action, logproba, _, invalid_action_masks[step] = agent.get_action(
                obs[step]
            )

        # Remind:
        # actions of shape (num_steps, num_envs, h*w, num_discrete_actions) = (256, 24, 256, 7)
        # action of shape (num_envs, h*w, num_discrete_actions) = (24, 256, 7)
        actions[step] = action

        # Remind:
        # logprobs of shape (num_steps, num_envs) = (256, 24)
        # logproba of shape (num_envs) = (24)
        logprobs[step] = logproba

        # TRY NOT TO MODIFY: execute the game and log data.
        # the real action adds the source units
        # of shape (num_envs, h*w, num_discrete_actions+1) = (24, 256, 8)
        real_action = torch.cat(
            [
                # of shape (24, 256, 1)
                torch.stack(
                    [
                        torch.arange(0, mapsize, device=device)
                        for i in range(envs.num_envs)
                    ]
                ).unsqueeze(2),
                # of shape (24, 256, 7)
                action,
            ],
            dim=2,
        )

        # at this point, the `real_action` has shape (num_envs, map_height*map_width, 8)
        # so as to predict an action for each cell in the map; this obviously include a
        # lot of invalid actions at cells for which no source units exist, so the rest of
        # the code removes these invalid actions to speed things up
        real_action = real_action.cpu().numpy()

        # Remind that: invalid_action_masks[step] is of shape (num_envs, mapsize, 79) = (24, 256, 79)
        # valid_actions is of shape (num_valid_actions, 8)
        valid_actions = real_action[
            invalid_action_masks[step][:, :, 0].bool().cpu().numpy()
        ]
        valid_actions_counts = (
            invalid_action_masks[step][:, :, 0].sum(1).long().cpu().numpy()
        )
        java_valid_actions = []
        valid_action_idx = 0
        for env_idx, valid_action_count in enumerate(valid_actions_counts):
            java_valid_action = []
            for c in range(valid_action_count):
                java_valid_action += [JArray(JInt)(valid_actions[valid_action_idx])]
                valid_action_idx += 1
            java_valid_actions += [JArray(JArray(JInt))(java_valid_action)]
        java_valid_actions = JArray(JArray(JArray(JInt)))(java_valid_actions)

        try:
            next_obs, rs, ds, infos = envs.step(java_valid_actions)
            next_obs = torch.Tensor(next_obs).to(device)
        except Exception as e:
            e.printStackTrace()
            raise
        rewards[step], next_done = torch.Tensor(rs).to(device), torch.Tensor(ds).to(
            device
        )

        for info in infos:
            if "episode" in info.keys():
                print(
                    f"global_step={global_step}, episode_reward={info['episode']['r']}"
                )
                writer.add_scalar(
                    "charts/episode_reward", info["episode"]["r"], global_step
                )
                for key in info["microrts_stats"]:
                    writer.add_scalar(
                        f"charts/episode_reward/{key}",
                        info["microrts_stats"][key],
                        global_step,
                    )
                break

    # bootstrap reward if not done. reached the batch limit
    with torch.no_grad():
        last_value = agent.get_value(next_obs.to(device)).reshape(1, -1)
        if args.gae:
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = last_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = (
                    rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                )
                advantages[t] = lastgaelam = (
                    delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
                )
            returns = advantages + values
        else:
            returns = torch.zeros_like(rewards).to(device)
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    next_return = last_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    next_return = returns[t + 1]
                returns[t] = rewards[t] + args.gamma * nextnonterminal * next_return
            advantages = returns - values

    # flatten the batch
    b_obs = obs.reshape((-1,) + envs.observation_space.shape)
    b_logprobs = logprobs.reshape(-1)
    b_actions = actions.reshape((-1,) + action_space_shape)
    b_advantages = advantages.reshape(-1)
    b_returns = returns.reshape(-1)
    b_values = values.reshape(-1)
    b_invalid_action_masks = invalid_action_masks.reshape((-1,) + invalid_action_shape)

    # Optimizaing the policy and value network
    inds = np.arange(
        args.batch_size,
    )
    for i_epoch_pi in range(args.update_epochs):
        np.random.shuffle(inds)
        for start in range(0, args.batch_size, args.minibatch_size):
            end = start + args.minibatch_size
            minibatch_ind = inds[start:end]
            mb_advantages = b_advantages[minibatch_ind]
            if args.norm_adv:
                mb_advantages = (mb_advantages - mb_advantages.mean()) / (
                    mb_advantages.std() + 1e-8
                )
            # raise
            _, newlogproba, entropy, _ = agent.get_action(
                b_obs[minibatch_ind],
                b_actions.long()[minibatch_ind],
                b_invalid_action_masks[minibatch_ind],
            )
            ratio = (newlogproba - b_logprobs[minibatch_ind]).exp()

            # Stats
            approx_kl = (b_logprobs[minibatch_ind] - newlogproba).mean()

            # Policy loss
            pg_loss1 = -mb_advantages * ratio
            pg_loss2 = -mb_advantages * torch.clamp(
                ratio, 1 - args.clip_coef, 1 + args.clip_coef
            )
            pg_loss = torch.max(pg_loss1, pg_loss2).mean()
            entropy_loss = entropy.mean()

            # Value loss
            new_values = agent.get_value(b_obs[minibatch_ind]).view(-1)
            if args.clip_vloss:
                v_loss_unclipped = (new_values - b_returns[minibatch_ind]) ** 2
                v_clipped = b_values[minibatch_ind] + torch.clamp(
                    new_values - b_values[minibatch_ind],
                    -args.clip_coef,
                    args.clip_coef,
                )
                v_loss_clipped = (v_clipped - b_returns[minibatch_ind]) ** 2
                v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                v_loss = 0.5 * v_loss_max.mean()
            else:
                v_loss = 0.5 * ((new_values - b_returns[minibatch_ind]) ** 2)

            loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
            optimizer.step()

    ## CRASH AND RESUME LOGIC:
    if args.prod_mode:
        if not os.path.exists(f"models/{experiment_name}"):
            os.makedirs(f"models/{experiment_name}")
            torch.save(agent.state_dict(), f"{wandb.run.dir}/agent.pt")
            wandb.save(f"agent.pt")
        else:
            if update % CHECKPOINT_FREQUENCY == 0:
                torch.save(agent.state_dict(), f"{wandb.run.dir}/agent.pt")

    # TRY NOT TO MODIFY: record rewards for plotting purposes
    writer.add_scalar(
        "charts/learning_rate", optimizer.param_groups[0]["lr"], global_step
    )
    writer.add_scalar("charts/update", update, global_step)
    writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
    writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
    writer.add_scalar("losses/entropy", entropy.mean().item(), global_step)
    writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
    if args.kle_stop or args.kle_rollback:
        writer.add_scalar("debug/pg_stop_iter", i_epoch_pi, global_step)
    writer.add_scalar(
        "charts/sps", int(global_step / (time.time() - start_time)), global_step
    )
    print("SPS:", int(global_step / (time.time() - start_time)))

envs.close()
writer.close()
