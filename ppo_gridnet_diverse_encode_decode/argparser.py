import os
from typing import Optional

from tap import Tap

# Common arguments
# parser.add_argument(
#     "--exp-name",
#     type=str,
#     default=os.path.basename(__file__).rstrip(".py"),
#     help="the name of this experiment",
# )
# parser.add_argument(
#     "--gym-id",
#     type=str,
#     default="MicrortsDefeatCoacAIShaped-v3",
#     help="the id of the gym environment",
# )
# parser.add_argument(
#     "--learning-rate",
#     type=float,
#     default=2.5e-4,
#     help="the learning rate of the optimizer",
# )
# parser.add_argument("--seed", type=int, default=1, help="seed of the experiment")
# parser.add_argument(
#     "--total-timesteps",
#     type=int,
#     default=100000000,
#     help="total timesteps of the experiments",
# )
# parser.add_argument(
#     "--torch-deterministic",
#     type=lambda x: bool(strtobool(x)),
#     default=True,
#     nargs="?",
#     const=True,
#     help="if toggled, `torch.backends.cudnn.deterministic=False`",
# )
# parser.add_argument(
#     "--cuda",
#     type=lambda x: bool(strtobool(x)),
#     default=True,
#     nargs="?",
#     const=True,
#     help="if toggled, cuda will not be enabled by default",
# )
# parser.add_argument(
#     "--prod-mode",
#     type=lambda x: bool(strtobool(x)),
#     default=False,
#     nargs="?",
#     const=True,
#     help="run the script in production mode and use wandb to log outputs",
# )
# parser.add_argument(
#     "--capture-video",
#     type=lambda x: bool(strtobool(x)),
#     default=False,
#     nargs="?",
#     const=True,
#     help="weather to capture videos of the agent performances (check out `videos` folder)",
# )
# parser.add_argument(
#     "--wandb-project-name",
#     type=str,
#     default="cleanRL",
#     help="the wandb's project name",
# )
# parser.add_argument(
#     "--wandb-entity",
#     type=str,
#     default=None,
#     help="the entity (team) of wandb's project",
# )

# # Algorithm specific arguments
# parser.add_argument(
#     "--n-minibatch", type=int, default=4, help="the number of mini batch"
# )
# parser.add_argument(
#     "--num-bot-envs",
#     type=int,
#     default=24,
#     help="the number of bot game environment; 16 bot envs measn 16 games",
# )
# parser.add_argument(
#     "--num-selfplay-envs",
#     type=int,
#     default=0,
#     help="the number of self play envs; 16 self play envs means 8 games",
# )
# parser.add_argument(
#     "--num-steps",
#     type=int,
#     default=256,
#     help="the number of steps per game environment",
# )
# parser.add_argument(
#     "--gamma", type=float, default=0.99, help="the discount factor gamma"
# )
# parser.add_argument(
#     "--gae-lambda",
#     type=float,
#     default=0.95,
#     help="the lambda for the general advantage estimation",
# )
# parser.add_argument(
#     "--ent-coef", type=float, default=0.01, help="coefficient of the entropy"
# )
# parser.add_argument(
#     "--vf-coef", type=float, default=0.5, help="coefficient of the value function"
# )
# parser.add_argument(
#     "--max-grad-norm",
#     type=float,
#     default=0.5,
#     help="the maximum norm for the gradient clipping",
# )
# parser.add_argument(
#     "--clip-coef",
#     type=float,
#     default=0.1,
#     help="the surrogate clipping coefficient",
# )
# parser.add_argument(
#     "--update-epochs", type=int, default=4, help="the K epochs to update the policy"
# )
# parser.add_argument(
#     "--kle-stop",
#     type=lambda x: bool(strtobool(x)),
#     default=False,
#     nargs="?",
#     const=True,
#     help="If toggled, the policy updates will be early stopped w.r.t target-kl",
# )
# parser.add_argument(
#     "--kle-rollback",
#     type=lambda x: bool(strtobool(x)),
#     default=False,
#     nargs="?",
#     const=True,
#     help="If toggled, the policy updates will roll back to previous policy if KL exceeds target-kl",
# )
# parser.add_argument(
#     "--target-kl",
#     type=float,
#     default=0.03,
#     help="the target-kl variable that is referred by --kl",
# )
# parser.add_argument(
#     "--gae",
#     type=lambda x: bool(strtobool(x)),
#     default=True,
#     nargs="?",
#     const=True,
#     help="Use GAE for advantage computation",
# )
# parser.add_argument(
#     "--norm-adv",
#     type=lambda x: bool(strtobool(x)),
#     default=True,
#     nargs="?",
#     const=True,
#     help="Toggles advantages normalization",
# )
# parser.add_argument(
#     "--anneal-lr",
#     type=lambda x: bool(strtobool(x)),
#     default=True,
#     nargs="?",
#     const=True,
#     help="Toggle learning rate annealing for policy and value networks",
# )
# parser.add_argument(
#     "--clip-vloss",
#     type=lambda x: bool(strtobool(x)),
#     default=True,
#     nargs="?",
#     const=True,
#     help="Toggles wheter or not to use a clipped loss for the value function, as per the paper.",
# )


class PPOGridNetDiverseEncodeDecodeArgumentParser(Tap):
    exp_name: Optional[str] = os.path.basename(__file__).rstrip(".py")
    gym_id: Optional[str] = "MicrortsDefeatCoacAIShaped-v3"
    learning_rate: Optional[float] = 2.5e-4
    seed: Optional[int] = 1
    total_timesteps: Optional[int] = 100_000_000
    torch_deterministic: Optional[bool] = True
    cuda: Optional[bool] = True
    prod_mode: Optional[
        bool
    ] = False  # run the script in production mode and use wandb to log outputs
    capture_video: Optional[bool] = False
    wandb_project_name: Optional[str] = "cleanRL"
    wandb_entity: Optional[str] = None
    n_minibatch: Optional[int] = 4
    num_bot_envs: Optional[
        int
    ] = 24  # the number of bot game environment; 16 bot envs means 16 games
    num_selfplay_envs: Optional[
        int
    ] = 0  # the number of self play envs; 16 self play envs means 8 games
    num_steps: Optional[int] = 256  # num of steps per game environment
    gamma: Optional[int] = 0.99  # discount factor gamma
    gae_lambda: Optional[float] = 0.95  # lambda for the general advantage estimation
    ent_coef: Optional[float] = 0.01  # coefficient of the entropy
    vf_coef: Optional[float] = 0.5  # coefficient of the value function
    max_grad_norm: Optional[float] = 0.5  # the maximum norm for the gradient clipping
    clip_coef: Optional[float] = 0.1  # the surrogate clipping coeeficient
    update_epochs: Optional[int] = 4  # the K epochs to update the policy
    kle_stop: Optional[bool] = False  # early stopped w.r.t target-kl
    kle_rollback: Optional[
        bool
    ] = False  # the policy updates will roll back to previous policy if KL exceeds target-kl
    target_kl: Optional[float] = 0.03  # target-kl variable that is referred by --kl
    gae: Optional[bool] = True  # Use GAE for advantage computation
    norm_adv: Optional[bool] = True  # Advantages normalization
    anneal_lr: Optional[
        bool
    ] = True  # Learning rate annealing for policy and value networks
    clip_vloss: Optional[
        bool
    ] = True  # whether or not to use a clipped loss for the value function

    # Additional arguments
    num_envs: Optional[int] = None

    # @Question: what is the difference between batch_size and minibatch_size
    batch_size: Optional[int] = None
    minibatch_size: Optional[int] = None

    def __init__(self):
        super().__init__(explicit_bool=True)
