"""
This script initiates the training of the RL Agent component after separately training the HAFED pipeline,
which includes the Feature Aggregator, Classifier and the Targeted State Updater (TSU).

During this phase:
- Pretrained weights for both the Feature Aggregator and TSU are loaded from their respective checkpoints.
- The RL Agent is trained to intelligently select patches based on the above models.


Commands
CAMELYON16
python step6_rl_training.py --config config/camelyon_rl_config.yml --seed 5  --log_dir LOG_DIR

TCGA
python step6_rl_training.py --config config/tcga_rl_config.yml --seed 1  --log_dir LOG_DIR

"""

import wandb
import argparse
import gc
import os
from pprint import pprint
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
import torchmetrics
import yaml
from sklearn.metrics import balanced_accuracy_score
from timm.utils import accuracy
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F


from architecture.transformer import HAFED
from datasets.datasets import build_HDF5_feat_dataset_2
from envs.WSI_cosine_env import WSICosineObservationEnv
from envs.WSI_env import WSIObservationEnv
from modules.fglobal_mlp import FGlobal
from rl_algorithms.ppo import Agent, Actor, Critic
from rl_algorithms.ppo import Agent
from rl_algorithms.ppo import TransformerActor as Actor
#from rl_algorithms.ppo import TransformerCritic as Critic
from step4_extract_intermediate_features import load_model
from utils.gpu_utils import check_gpu_availability
from utils.utils import MetricLogger, SmoothedValue, adjust_learning_rate
from utils.utils import save_policy_model, Struct, set_seed


def get_arguments():
    parser = argparse.ArgumentParser('RL training', add_help=False)
    parser.add_argument('--config', default= None, help='path to config file')
    parser.add_argument('--seed', type=int, default=4, help='set the random seed')
    parser.add_argument('--classifier_arch', default='hafed', choices=['hafed'], help='choice of architecture for HACMIL')
    parser.add_argument('--exp_name', type=str, default='DEBUG', help='name of the exp')
    parser.add_argument('--logs', default='enabled', choices=['enabled', 'disabled'], type=str, help='flag to save logs')
    parser.add_argument('--log_dir', default=None, type=str, help='Log dir path to save model and logs')
    parser.add_argument('--restart', action='store_true')
    parser.add_argument('--rl_ckpt_path', default=None, type=str, help='Log dir path to save model and logs')
    parser.add_argument('--lambda_global', type=float, default=0.25, help='set the random seed')
    parser.add_argument('--lambda_diversity', type=float, default=0.25, help='set the random seed')
    parser.add_argument('--lambda_coverage', type=float, default=0.25, help='set the random seed')
    parser.add_argument('--lambda_ssl', type=float, default=0.25, help='set the random seed')
    parser.add_argument('--frac_visit', type=float, default=0.2)

    args = parser.parse_args()

    # Adding Device Details
    gpus = check_gpu_availability(10, 1, [])
    print(f"occupied {gpus}")
    args.device = torch.device(f"cuda:{gpus[0]}")

    return args

def compute_rtgs(batch_rews, conf):
    """
        Compute the Reward-To-Go of each timestep in a batch given the rewards.

        Parameters:
            batch_rews - the rewards in a batch, Shape: (number of episodes, number of timesteps per episode)

        Return:
            batch_rtgs - the rewards to go, Shape: (number of timesteps in batch)
    """
    # The rewards-to-go (rtg) per episode per batch to return.
    # The shape will be (num timesteps per episode)
    batch_rtgs = []

    # Iterate through each episode
    for ep_rews in reversed(batch_rews):

        discounted_reward = 0 # The discounted reward so far

        # Iterate through all rewards in the episode. We go backwards for smoother calculation of each
        # discounted return (think about why it would be harder starting from the beginning)
        for rew in reversed(ep_rews):
            discounted_reward = rew + discounted_reward * conf.gamma
            batch_rtgs.insert(0, discounted_reward)

    # Convert the rewards-to-go into a tensor
    batch_rtgs = torch.tensor(batch_rtgs, dtype=torch.float)

    return batch_rtgs


def gae(rewards, values, episode_ends, gamma, lam):
    """Compute generalized advantage estimate.
        rewards: a list of rewards at each step.
        values: the value estimate of the state at each step.
        episode_ends: an array of the same shape as rewards, with a 1 if the
            episode ended at that step and a 0 otherwise.
        gamma: the discount factor.
        lam: the GAE lambda parameter.
    """
    # Invert episode_ends to have 0 if the episode ended and 1 otherwise
    episode_ends = (episode_ends * -1) + 1

    N = rewards.shape[0]
    T = rewards.shape[1]
    gae_step = np.zeros((N, ))
    advantages = np.zeros((N, T))

    for t in reversed(range(T)):
    # for t in reversed(range(T - 1)):
        # First compute delta, which is the one-step TD error
        next_values = values[:, t + 1] if (t + 1) < T else 0.0
        delta = rewards[:, t] + gamma * next_values * episode_ends[:, t] - values[:, t]
        # delta = rewards[:, t] + gamma * values[:, t + 1] * episode_ends[:, t] - values[:, t]
        # Then compute the current step's GAE by discounting the previous step
        # of GAE, resetting it to zero if the episode ended, and adding this
        # step's delta
        gae_step = delta + gamma * lam * episode_ends[:, t] * gae_step
        # And store it
        advantages[:, t] = gae_step
    return advantages


def load_policy_model(model, actor_optimizer, critic_optimizer, load_path, device="cpu"):
    # Load the checkpoint
    checkpoint = torch.load(load_path, map_location=device)

    # Load model weights
    model.load_state_dict(checkpoint['model'])

    # Load optimizer states
    actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
    critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])

    # Get epoch number and config
    epoch = checkpoint['epoch']
    config = checkpoint['config']

    print(f"Model loaded from {load_path} at epoch {epoch}")

    return model, actor_optimizer, critic_optimizer, epoch, config
def load_actor_only(model, load_path, device="cpu"):

    checkpoint = torch.load(load_path, map_location=device)

    full_state_dict = checkpoint["model"]

    actor_state_dict = {
        key.replace("actor.", "", 1): value
        for key, value in full_state_dict.items()
        if key.startswith("actor.")
    }

    if len(actor_state_dict) == 0:
        raise RuntimeError(
            f"No actor.* parameters were found in checkpoint: {load_path}"
        )

    missing_keys, unexpected_keys = model.actor.load_state_dict(
        actor_state_dict,
        strict=True,
    )

    return model

def main(random_seed=11):
    # getting and config file
    args = get_arguments()


    with open(args.config, 'r') as ymlfile:
        c = yaml.load(ymlfile, Loader=yaml.FullLoader)
        c.update(vars(args))
        conf = Struct(**c)
        conf.log_dir = os.path.join(
            conf.log_dir,
            f"explore_frac_{conf.frac_visit}_seed_{conf.seed}",
        )

    conf.writer = SummaryWriter(log_dir=os.path.join(conf.log_dir, "logs", conf.exp_name))

    hyparams = {
        'dataset': conf.dataset,
        'pretrain': conf.pretrain,
        'classifier_arch': conf.classifier_arch,
        'lr': conf.lr,
        'seed': conf.seed,
        'frac_visit': conf.frac_visit,
        'gamma': conf.gamma,
        'clip': conf.clip,
        'num_envs': conf.num_envs,
        'num_epochs_on_single_roll_out': conf.num_epochs_on_single_roll_out,
        'only_ce_as_reward': conf.only_ce_as_reward,
        'lambda': conf.lam,
        'use_gae': conf.use_gae,
        'max_gradient_norm': conf.max_grad_norm,
        'entropy_coef': conf.entropy_coef,
        'use_entropy_loss': conf.use_entropy_loss,
        'lambda_ssl': conf.lambda_ssl,
        'lambda_diversity': conf.lambda_diversity,
        'lambda_coverage': conf.lambda_coverage,
        'global_reward_method': conf.global_reward_method,
        'global_reward_k': conf.global_reward_k,
        'reward_scale': conf.reward_scale,
        'lambda_global': conf.lambda_global,
    }
    hyparams['fraction of visit'] = conf.frac_visit
    hyparams['num_envs'] = conf.num_envs
    hyparams_text = "\n".join([f"**{key}**: {value}" for key, value in hyparams.items()])
    conf.writer.add_text("Hyperparameters", hyparams_text)


    ckpt_dir = os.path.join(conf.log_dir, "models", conf.exp_name)
    os.makedirs(ckpt_dir, exist_ok=True)  # Create the 'ckpt' directory if it doesn't exist

    print("Used config:");
    pprint(vars(conf));

    wandb.init(
            project="RL-KNIFE-SSL",
            name=conf.exp_name,
            config=vars(conf),
            mode="offline", 
        )

    # Loading seed
    set_seed(args.seed)


    # dataloaders
    train_data, val_data, test_data = build_HDF5_feat_dataset_2(conf.level1_path, conf.level3_path, conf)
    train_loader = DataLoader(train_data, batch_size=conf.B, shuffle=True, num_workers=conf.n_worker, pin_memory=conf.pin_memory, drop_last=True)
    val_loader = DataLoader(val_data, batch_size=conf.B, shuffle=False, num_workers=conf.n_worker, pin_memory=conf.pin_memory, drop_last=False)
    test_loader = DataLoader(test_data, batch_size=conf.B, shuffle=False, num_workers=conf.n_worker, pin_memory=conf.pin_memory, drop_last=False)

    # Loading HAFED
    #classifier_dict, _, _, _ = load_model(ckpt_path=conf.classifier_ckpt_path, args= args)
    #classifier_conf = SimpleNamespace(**config)

    classifier = None
    
    # Loading TSU
    fglobal_dict = torch.load(conf.mlp_fglobal_ckpt, map_location=conf.device)
    fglobal = FGlobal(ip_dim=384 * 3, op_dim=384).to(conf.device)
    fglobal.load_state_dict(fglobal_dict['model'])
    fglobal.eval()

    # Loading RL Agent ----> Actor / Critic
    actor = Actor(conf=conf)
    critic = Critic(conf=conf)
    model = Agent(actor, critic, conf).to(conf.device)
    actor_optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, actor.parameters()), lr=conf.lr, weight_decay=conf.wd)
    critic_optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, critic.parameters()), lr=conf.lr, weight_decay=conf.wd)


    best_state = {
            "epoch": -1,
            "val_final_ssl_loss": float("inf"),
            "val_reward": float("-inf"),
            "test_final_ssl_loss": float("inf"),
            "test_reward": float("-inf"),
        }
    start_epoch = 0

    train_epoch = conf.train_epoch
    epoch = -1
    val_loss, val_reward = evaluate_foundation(
            model,
            fglobal,
            val_loader,
            "Val",
            conf.device,
            epoch,
            conf,
        )
    test_loss, test_reward = evaluate_foundation(
            model,
            fglobal,
            test_loader,
            "Test",
            conf.device,
            epoch,
            conf,
        )
        
    log_dict = {
            "val_loss": val_loss,
            "val_reward": val_reward,
            "test_loss": test_loss,
            "test_reward": test_reward,
            "epoch": epoch,
        }

    wandb.log(log_dict)

    for epoch in range(start_epoch, train_epoch):
        train_one_epoch_foundation(
                model,
                fglobal,
                train_loader,
                actor_optimizer,
                critic_optimizer,
                conf.device,
                epoch,
                conf,
            )

        val_loss, val_reward = evaluate_foundation(
            model,
            fglobal,
            val_loader,
            "Val",
            conf.device,
            epoch,
            conf,
        )
        test_loss, test_reward = evaluate_foundation(
            model,
            fglobal,
            test_loader,
            "Test",
            conf.device,
            epoch,
            conf,
        )
        
        log_dict = {
            "val_loss": val_loss,
            "val_reward": val_reward,
            "test_loss": test_loss,
            "test_reward": test_reward,
            "epoch": epoch,
        }

        wandb.log(log_dict)

        # The policy is now optimized for exploration reward, so checkpoint
        # selection must maximize validation reward rather than minimize the
        # old full-view cosine loss. The SSL loss is retained only as a
        # diagnostic metric.
        if val_reward > best_state["val_reward"]:
            best_state["epoch"] = epoch
            best_state["val_final_ssl_loss"] = val_loss
            best_state["val_reward"] = val_reward
            best_state["test_final_ssl_loss"] = test_loss
            best_state["test_reward"] = test_reward
            save_policy_model(
                conf=conf,
                model=model,
                actor_optimizer=actor_optimizer,
                critic_optimizer=critic_optimizer,
                epoch=epoch,
                save_path=os.path.join(ckpt_dir, "checkpoint-best.pt"),
            )


    print('\n')

    save_policy_model(conf=conf, model=model, actor_optimizer=actor_optimizer, critic_optimizer=critic_optimizer,
                      epoch=epoch, save_path=os.path.join(ckpt_dir, 'checkpoint-last.pt'))

    print("Results on best epoch:")
    print(best_state)
    best_state_text = "\n".join([f"{key}: {value}" for key, value in best_state.items()])
    conf.writer.add_text("Best Model State", best_state_text, global_step=best_state["epoch"])
    conf.writer.close()

def ssl_state_loss(state, hr_features):
    """
    state:       [1, N, D] or [N, D]
    hr_features: [N, D]

    Lower is better.
    """
    if state.dim() == 3:
        state = state.squeeze(0)

    return 1.0 - F.cosine_similarity(
        F.normalize(state, dim=-1),
        F.normalize(hr_features, dim=-1),
        dim=-1
    ).mean() 




def _normalize_patch_features(features):
    """Validate and L2-normalize a [N, D] patch-feature matrix."""
    if features.dim() == 3 and features.shape[0] == 1:
        features = features.squeeze(0)

    if features.dim() != 2:
        raise ValueError(
            f"Expected patch features with shape [N, D], got {tuple(features.shape)}"
        )

    if features.shape[0] == 0:
        raise ValueError("Cannot calculate rewards for an empty patch set.")

    if not torch.isfinite(features).all():
        raise ValueError("Patch features contain NaN or Inf values.")

    return F.normalize(features.float(), p=2, dim=-1, eps=1e-8)


def _scaled_cosine_matrix(normalized_a, normalized_b):
    """Cosine similarity mapped from [-1, 1] to [0, 1]."""
    cosine = normalized_a @ normalized_b.transpose(-1, -2)
    return ((cosine + 1.0) / 2.0).clamp(0.0, 1.0)


@torch.no_grad()
def calculate_global_rank_rewards(
    combined_hr_features,
    method="knn_mean",
    k=20,
):
    """
    Calculate one slide-relative global uniqueness reward per patch.

    Returns
    -------
    rank_rewards: [N], from 0 (least unique) to 1 (most unique)
    raw_uniqueness: [N], absolute uniqueness in [0, 1]
    """
    features = _normalize_patch_features(combined_hr_features)
    n = features.shape[0]

    if n == 1:
        ones = torch.ones(1, device=features.device, dtype=features.dtype)
        return ones, ones

    similarity = _scaled_cosine_matrix(features, features)

    if method == "knn_mean":
        similarity_no_self = similarity.clone()
        similarity_no_self.fill_diagonal_(float("-inf"))
        effective_k = min(max(int(k), 1), n - 1)
        local_similarity = torch.topk(
            similarity_no_self,
            k=effective_k,
            dim=1,
            largest=True,
        ).values.mean(dim=1)

    elif method == "global_mean":
        similarity_no_self = similarity.clone()
        similarity_no_self.fill_diagonal_(0.0)
        local_similarity = similarity_no_self.sum(dim=1) / float(n - 1)

    elif method == "max":
        similarity_no_self = similarity.clone()
        similarity_no_self.fill_diagonal_(float("-inf"))
        local_similarity = similarity_no_self.max(dim=1).values

    else:
        raise ValueError(
            "global_reward_method must be one of: "
            "'knn_mean', 'global_mean', or 'max'."
        )

    raw_uniqueness = (1.0 - local_similarity).clamp(0.0, 1.0)

    sorted_indices = torch.argsort(raw_uniqueness, descending=False)
    integer_ranks = torch.empty(n, device=features.device, dtype=torch.long)
    integer_ranks[sorted_indices] = torch.arange(n, device=features.device)
    rank_rewards = integer_ranks.float() / float(n - 1)

    return rank_rewards, raw_uniqueness


@torch.no_grad()
def calculate_observed_diversity_reward(
    combined_hr_features,
    action_index,
    observed_indices,
    reduction="max",
    first_action_reward=0.0,
):
    """
    Reward a candidate for being different from already selected patches.
    The recommended reduction is 'max', which penalizes any near-duplicate.
    """
    features = _normalize_patch_features(combined_hr_features)
    action_index = int(action_index.item()) if torch.is_tensor(action_index) else int(action_index)

    if len(observed_indices) == 0:
        return torch.tensor(
            first_action_reward,
            device=features.device,
            dtype=features.dtype,
        )

    observed = torch.as_tensor(
        observed_indices,
        device=features.device,
        dtype=torch.long,
    ).flatten().unique()

    action_feature = features[action_index : action_index + 1]
    observed_features = features[observed]
    similarities = _scaled_cosine_matrix(
        action_feature,
        observed_features,
    ).squeeze(0)

    if reduction == "max":
        redundancy = similarities.max()
    elif reduction == "mean":
        redundancy = similarities.mean()
    else:
        raise ValueError("reduction must be either 'max' or 'mean'.")

    return (1.0 - redundancy).clamp(0.0, 1.0)


@torch.no_grad()
def calculate_coverage_reward(
    combined_hr_features,
    action_index,
    current_best_coverage=None,
):
    """
    Calculate the relative marginal facility-location coverage gain.

    The absolute coverage is the mean similarity of each WSI region to its
    closest selected representative. Relative gain divides the new coverage
    by the remaining uncovered mass, keeping this component approximately
    in [0, 1] and making it comparable with the other reward components.
    """
    features = _normalize_patch_features(combined_hr_features)
    n = features.shape[0]
    action_index = int(action_index.item()) if torch.is_tensor(action_index) else int(action_index)

    if current_best_coverage is None:
        current_best_coverage = torch.zeros(
            n,
            device=features.device,
            dtype=features.dtype,
        )
    else:
        current_best_coverage = current_best_coverage.to(
            device=features.device,
            dtype=features.dtype,
        ).flatten()
        if current_best_coverage.shape != (n,):
            raise ValueError(
                f"current_best_coverage must have shape [{n}], "
                f"got {tuple(current_best_coverage.shape)}"
            )

    candidate_feature = features[action_index : action_index + 1]
    candidate_coverage = _scaled_cosine_matrix(
        features,
        candidate_feature,
    ).squeeze(1)

    updated_best_coverage = torch.maximum(
        current_best_coverage,
        candidate_coverage,
    )

    coverage_before = current_best_coverage.mean()
    coverage_after = updated_best_coverage.mean()
    absolute_gain = (coverage_after - coverage_before).clamp_min(0.0)
    remaining_uncovered = (1.0 - coverage_before).clamp_min(1e-8)
    relative_gain = (absolute_gain / remaining_uncovered).clamp(0.0, 1.0)

    return (
        relative_gain,
        updated_best_coverage,
        coverage_before,
        coverage_after,
        absolute_gain,
    )


@torch.no_grad()
def calculate_exploration_reward(
    combined_hr_features,
    action_index,
    observed_indices,
    global_rank_rewards,
    current_best_coverage,
    lambda_global=0.10,
    lambda_diversity=0.30,
    lambda_coverage=0.60,
    reward_scale=1.0,
):
    """Combine global uniqueness, selected-set diversity, and coverage."""
    action = int(action_index.item()) if torch.is_tensor(action_index) else int(action_index)

    global_reward = global_rank_rewards[action]

    diversity_reward = calculate_observed_diversity_reward(
            combined_hr_features=combined_hr_features,
            action_index=action,
            observed_indices=observed_indices,
            reduction="max",
        )

    (
        coverage_reward,
        updated_best_coverage,
        coverage_before,
        coverage_after,
        absolute_coverage_gain,
    ) = calculate_coverage_reward(
        combined_hr_features=combined_hr_features,
        action_index=action,
        current_best_coverage=current_best_coverage,
    )

    total_reward = reward_scale * (
        lambda_global * global_reward
        + lambda_diversity * diversity_reward
        + lambda_coverage * coverage_reward
    )

    return total_reward, {
        "global_reward": global_reward,
        "diversity_reward": diversity_reward,
        "coverage_reward": coverage_reward,
        "absolute_coverage_gain": absolute_coverage_gain,
        "coverage_before": coverage_before,
        "coverage_after": coverage_after,
        "updated_best_coverage": updated_best_coverage,
    }

def train_one_epoch_foundation(
    model,
    fglobal,
    data_loader,
    actor_optimizer,
    critic_optimizer,
    device,
    epoch,
    conf,
):
    model.train()
    fglobal.eval()

    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter(
        "actor_lr",
        SmoothedValue(window_size=1, fmt="{value:.6f}")
    )
    metric_logger.add_meter(
        "critic_lr",
        SmoothedValue(window_size=1, fmt="{value:.6f}")
    )

    header = "Foundation Epoch: [{}]".format(epoch)
    print_freq = 100

    epoch_actor_loss = 0.0
    epoch_critic_loss = 0.0

    for data_it, data in enumerate(
        metric_logger.log_every(
            data_loader,
            print_freq,
            header
        )
    ):

        # ============================================================
        # Data
        # ============================================================

        hr_features = data["hr"][0].to(
            device,
            dtype=torch.float32
        )

        initial_state = data["lr"].to(
            device,
            dtype=torch.float32
        ).clone()

        # ------------------------------------------------------------
        # Global uniqueness ranks are fixed for the whole WSI.
        # ------------------------------------------------------------

        global_rank_rewards, _ = calculate_global_rank_rewards(
            combined_hr_features=hr_features,
            method=conf.global_reward_method,
            k=conf.global_reward_k,
        )

        # ============================================================
        # PPO rollout storage
        # ============================================================

        batch_obs = []
        batch_acts = []
        batch_log_probs = []
        batch_rews = []
        batch_lens = []
        dones = []

        # Reward-component diagnostics
        component_ssl = []
        component_global = []
        component_diversity = []
        component_coverage = []

        final_coverages = []
        final_pairwise_similarities = []
        final_ssl_losses = []

        # ============================================================
        # Collect on-policy trajectories
        # ============================================================

        with torch.no_grad():

            for env_idx in range(conf.num_envs):

                # ----------------------------------------------------
                # IMPORTANT:
                # This is now the SSL version of the SAME environment
                # transition dynamics used by SASHA.
                # ----------------------------------------------------

                env = WSICosineObservationEnv(
                    lr_features=initial_state.clone(),
                    hr_features=hr_features,
                    conf=conf,
                )

                env.reset()

                state = initial_state.clone()

                ep_rews = []
                visited_patch_ids = []

                current_best_coverage = None

                done = False
                ep_t = 0

                # ----------------------------------------------------
                # EXACTLY like SASHA:
                # let the ENVIRONMENT control episode termination.
                # ----------------------------------------------------

                while not done:

                    # ------------------------------------------------
                    # Store state BEFORE action
                    # ------------------------------------------------

                    batch_obs.append(
                        state.clone()
                    )

                    # ------------------------------------------------
                    # Sample action from current policy
                    # ------------------------------------------------

                    action, log_prob, entropy = model.get_action(
                        state,
                        visited_patch_ids,
                        is_eval=False,
                    )

                    action_idx = int(action.item())

                    # ------------------------------------------------
                    # Check whether environment is already in the
                    # "all patches covered" no-op state.
                    #
                    # In SASHA, actions after this point do not alter
                    # the state; therefore they should NOT receive
                    # artificial SSL/exploration reward.
                    # ------------------------------------------------

                    state_was_fully_covered = (
                        env.visited_all_patches
                    )

                    # ------------------------------------------------
                    # SSL loss BEFORE transition
                    # ------------------------------------------------

                    loss_before = ssl_state_loss(
                        state,
                        hr_features,
                    )

                    # ------------------------------------------------
                    # Exploration reward BEFORE transition
                    #
                    # Only meaningful if the action is actually going
                    # to be processed by the environment.
                    # ------------------------------------------------

                    if not state_was_fully_covered:

                        exploration_reward, reward_details = (
                            calculate_exploration_reward(
                                combined_hr_features=hr_features,
                                action_index=action,
                                observed_indices=visited_patch_ids,
                                global_rank_rewards=global_rank_rewards,
                                current_best_coverage=current_best_coverage,
                                lambda_global=conf.lambda_global,
                                lambda_diversity=conf.lambda_diversity,
                                lambda_coverage=conf.lambda_coverage,
                                reward_scale=conf.reward_scale,
                            )
                        )

                        current_best_coverage = (
                            reward_details[
                                "updated_best_coverage"
                            ]
                        )

                        global_reward = (
                            reward_details[
                                "global_reward"
                            ]
                        )

                        diversity_reward = (
                            reward_details[
                                "diversity_reward"
                            ]
                        )

                        coverage_reward = (
                            reward_details[
                                "coverage_reward"
                            ]
                        )

                    else:

                        # Environment is already effectively frozen.
                        exploration_reward = torch.tensor(
                            0.0,
                            dtype=torch.float32,
                            device=device,
                        )

                        global_reward = torch.tensor(
                            0.0,
                            dtype=torch.float32,
                            device=device,
                        )

                        diversity_reward = torch.tensor(
                            0.0,
                            dtype=torch.float32,
                            device=device,
                        )

                        coverage_reward = torch.tensor(
                            0.0,
                            dtype=torch.float32,
                            device=device,
                        )

                    # ------------------------------------------------
                    # Environment transition
                    # ------------------------------------------------

                    new_state, _, done = env.step(
                        action=action,
                        state_update_net=fglobal,
                        device=device,
                    )

                    # ------------------------------------------------
                    # SSL loss AFTER transition
                    # ------------------------------------------------

                    loss_after = ssl_state_loss(
                        new_state,
                        hr_features,
                    )

                    # ------------------------------------------------
                    # SSL reward
                    #
                    # r_ssl =
                    # reward_scale * lambda_ssl *
                    # (L_before - L_after)
                    #
                    # Since L = 10 * (1 - cosine),
                    # positive reward means state became closer to HR.
                    # ------------------------------------------------

                    if not state_was_fully_covered:

                        ssl_reward = (
                            conf.reward_scale
                            * conf.lambda_ssl
                            * (
                                loss_before
                                - loss_after
                            )
                        )

                    else:

                        ssl_reward = torch.tensor(
                            0.0,
                            dtype=torch.float32,
                            device=device,
                        )

                    # ------------------------------------------------
                    # Final foundation reward
                    # ------------------------------------------------

                    reward = (
                        exploration_reward
                        + ssl_reward
                    )

                    # ------------------------------------------------
                    # Store transition
                    # ------------------------------------------------

                    ep_rews.append(
                        float(reward.item())
                    )

                    batch_acts.append(
                        action_idx
                    )

                    batch_log_probs.append(
                        float(log_prob.item())
                    )

                    dones.append(
                        done
                    )

                    # Diagnostics
                    component_ssl.append(
                        float(ssl_reward.item())
                    )

                    component_global.append(
                        float(global_reward.item())
                    )

                    component_diversity.append(
                        float(diversity_reward.item())
                    )

                    component_coverage.append(
                        float(coverage_reward.item())
                    )

                    # ------------------------------------------------
                    # Keep SASHA behavior:
                    # explicitly chosen action is added to policy's
                    # visited list.
                    # ------------------------------------------------

                    visited_patch_ids.append(
                        action_idx
                    )

                    state = new_state

                    ep_t += 1

                # ====================================================
                # End of one rollout/environment
                # ====================================================

                batch_lens.append(
                    ep_t
                )

                batch_rews.append(
                    ep_rews
                )

                # ----------------------------------------------------
                # Final coverage diagnostic
                # ----------------------------------------------------

                if current_best_coverage is not None:

                    final_coverages.append(
                        float(
                            current_best_coverage
                            .mean()
                            .item()
                        )
                    )

                else:

                    final_coverages.append(
                        0.0
                    )

                # ----------------------------------------------------
                # Final selected-patch pairwise cosine
                # ----------------------------------------------------

                if len(visited_patch_ids) > 1:

                    selected = hr_features[
                        torch.as_tensor(
                            visited_patch_ids,
                            device=device,
                            dtype=torch.long,
                        )
                    ]

                    selected = F.normalize(
                        selected,
                        dim=-1,
                    )

                    selected_similarity = (
                        selected @ selected.T
                    )

                    upper = torch.triu_indices(
                        selected.shape[0],
                        selected.shape[0],
                        offset=1,
                        device=device,
                    )

                    final_pairwise_similarities.append(
                        float(
                            selected_similarity[
                                upper[0],
                                upper[1],
                            ]
                            .mean()
                            .item()
                        )
                    )

                else:

                    final_pairwise_similarities.append(
                        0.0
                    )

                # ----------------------------------------------------
                # Final SSL-state diagnostic
                # ----------------------------------------------------

                final_ssl_losses.append(
                    float(
                        ssl_state_loss(
                            state,
                            hr_features,
                        ).item()
                    )
                )

        # ============================================================
        # Convert rollout to tensors
        # ============================================================

        batch_obs = torch.stack(
            batch_obs
        ).squeeze(1)

        batch_acts = torch.tensor(
            batch_acts,
            dtype=torch.long,
            device=device,
        )

        batch_log_probs = torch.tensor(
            batch_log_probs,
            dtype=torch.float32,
            device=device,
        )

        # Monte-Carlo RTG.
        # Used directly when GAE=False.
        batch_rtgs = compute_rtgs(
            batch_rews,
            conf,
        ).to(
            device
        ).flatten()

        # ============================================================
        # Old value predictions
        #
        # This block is still inside no-grad conceptually:
        # these values are only used to construct GAE targets.
        # ============================================================

        with torch.no_grad():

            V_old, _, _ = model.evaluate(
                batch_obs,
                batch_acts,
            )

            V_old = V_old.flatten()

            # ========================================================
            # GAE
            # ========================================================

            if conf.use_gae:

                V_numpy = (
                    V_old
                    .view(
                        conf.num_envs,
                        -1
                    )
                    .cpu()
                    .numpy()
                )

                dones_numpy = np.asarray(
                    dones,
                    dtype=np.float32,
                ).reshape(
                    conf.num_envs,
                    -1,
                )

                rews_numpy = np.asarray(
                    batch_rews,
                    dtype=np.float32,
                ).reshape(
                    conf.num_envs,
                    -1,
                )

                A_k = gae(
                    rewards=rews_numpy,
                    values=V_numpy,
                    episode_ends=dones_numpy,
                    gamma=conf.gamma,
                    lam=conf.lam,
                )

                A_k = torch.tensor(
                    A_k,
                    dtype=torch.float32,
                    device=device,
                ).flatten()

                # ----------------------------------------------------
                # Critic target = GAE lambda return
                #
                # R_lambda = A_GAE + V_old
                # ----------------------------------------------------

                value_targets = (
                    A_k
                    + V_old
                )

            else:

                # ----------------------------------------------------
                # Standard Monte-Carlo target
                # ----------------------------------------------------

                value_targets = (
                    batch_rtgs
                )

                A_k = (
                    batch_rtgs
                    - V_old
                )

            # --------------------------------------------------------
            # Normalize ONLY advantages used by actor.
            #
            # Do this AFTER constructing value_targets.
            # --------------------------------------------------------

            A_k = (
                A_k
                - A_k.mean()
            ) / (
                A_k.std()
                + 1e-8
            )

        # ============================================================
        # PPO optimization
        # ============================================================

        for _ in range(
            conf.num_epochs_on_single_roll_out
        ):

            V, curr_log_probs, entropy = (
                model.evaluate(
                    batch_obs,
                    batch_acts,
                )
            )

            V = V.flatten()

            # --------------------------------------------------------
            # PPO probability ratio
            # --------------------------------------------------------

            ratios = torch.exp(
                curr_log_probs
                - batch_log_probs
            )

            # --------------------------------------------------------
            # PPO clipped objective
            # --------------------------------------------------------

            surr1 = (
                ratios
                * A_k
            )

            surr2 = (
                torch.clamp(
                    ratios,
                    1 - conf.clip,
                    1 + conf.clip,
                )
                * A_k
            )

            actor_loss = (
                -torch.min(
                    surr1,
                    surr2,
                )
            ).mean()

            # --------------------------------------------------------
            # Critic
            # --------------------------------------------------------

            critic_loss = nn.MSELoss()(
                V,
                value_targets,
            )

            # --------------------------------------------------------
            # Entropy bonus
            # --------------------------------------------------------

            entropy_loss = (
                entropy.mean()
            )

            if conf.use_entropy_loss:

                actor_loss -= (
                    conf.entropy_coef
                    * entropy_loss
                )

            # ========================================================
            # Actor update
            # ========================================================

            actor_optimizer.zero_grad()

            actor_loss.backward()

            nn.utils.clip_grad_norm_(
                model.actor.parameters(),
                conf.max_grad_norm,
            )

            actor_optimizer.step()

            # ========================================================
            # Critic update
            # ========================================================

            critic_optimizer.zero_grad()

            critic_loss.backward()

            nn.utils.clip_grad_norm_(
                model.critic.parameters(),
                conf.max_grad_norm,
            )

            critic_optimizer.step()

            epoch_actor_loss += (
                actor_loss.item()
            )

            epoch_critic_loss += (
                critic_loss.item()
            )

        # ============================================================
        # Diagnostics / logging
        # ============================================================

        mean_episode_reward = float(
            np.mean(
                [
                    np.sum(r)
                    for r in batch_rews
                ]
            )
        )

        mean_final_ssl = float(
            np.mean(
                final_ssl_losses
            )
        )

        mean_final_coverage = float(
            np.mean(
                final_coverages
            )
        )

        mean_pairwise_cosine = float(
            np.mean(
                final_pairwise_similarities
            )
        )

        mean_global_reward = float(
            np.mean(
                component_global
            )
        )

        mean_diversity_reward = float(
            np.mean(
                component_diversity
            )
        )

        mean_coverage_reward = float(
            np.mean(
                component_coverage
            )
        )

        mean_ssl_reward = float(
            np.mean(
                component_ssl
            )
        )

        # ------------------------------------------------------------
        # Metric logger
        # ------------------------------------------------------------

        metric_logger.update(
            actor_lr=actor_optimizer.param_groups[0]["lr"]
        )

        metric_logger.update(
            critic_lr=critic_optimizer.param_groups[0]["lr"]
        )

        metric_logger.update(
            actor_loss=actor_loss.item()
        )

        metric_logger.update(
            critic_loss=critic_loss.item()
        )

        metric_logger.update(
            reward=mean_episode_reward
        )

        metric_logger.update(
            global_reward=mean_global_reward
        )

        metric_logger.update(
            diversity_reward=mean_diversity_reward
        )

        metric_logger.update(
            coverage_reward=mean_coverage_reward
        )

        metric_logger.update(
            ssl_reward=mean_ssl_reward
        )

        metric_logger.update(
            final_coverage=mean_final_coverage
        )

        metric_logger.update(
            selected_pairwise_cosine=mean_pairwise_cosine
        )

        metric_logger.update(
            final_ssl_loss=mean_final_ssl
        )

        # ============================================================
        # TensorBoard
        # ============================================================

        if conf.logs != "disabled":

            step = (
                data_it
                + epoch * len(data_loader)
            )

            conf.writer.add_scalar(
                "Foundation/actor_loss",
                actor_loss.item(),
                step,
            )

            conf.writer.add_scalar(
                "Foundation/critic_loss",
                critic_loss.item(),
                step,
            )

            conf.writer.add_scalar(
                "Foundation/reward",
                mean_episode_reward,
                step,
            )

            conf.writer.add_scalar(
                "Foundation/global_reward",
                mean_global_reward,
                step,
            )

            conf.writer.add_scalar(
                "Foundation/diversity_reward",
                mean_diversity_reward,
                step,
            )

            conf.writer.add_scalar(
                "Foundation/coverage_reward",
                mean_coverage_reward,
                step,
            )

            conf.writer.add_scalar(
                "Foundation/ssl_reward",
                mean_ssl_reward,
                step,
            )

            conf.writer.add_scalar(
                "Foundation/final_coverage",
                mean_final_coverage,
                step,
            )

            conf.writer.add_scalar(
                "Foundation/selected_pairwise_cosine",
                mean_pairwise_cosine,
                step,
            )

            conf.writer.add_scalar(
                "Foundation/final_ssl_loss_diagnostic",
                mean_final_ssl,
                step,
            )

        # ============================================================
        # WandB
        # ============================================================

        wandb.log(
            {
                "train/reward":
                    mean_episode_reward,

                "train/global_reward":
                    mean_global_reward,

                "train/diversity_reward":
                    mean_diversity_reward,

                "train/coverage_reward":
                    mean_coverage_reward,

                "train/ssl_reward":
                    mean_ssl_reward,

                "train/final_coverage":
                    mean_final_coverage,

                "train/selected_pairwise_cosine":
                    mean_pairwise_cosine,

                "train/final_ssl_loss_diagnostic":
                    mean_final_ssl,

                "train/actor_loss":
                    actor_loss.item(),

                "train/critic_loss":
                    critic_loss.item(),

                "train/step":
                    data_it
                    + epoch * len(data_loader),
            }
        )

        # ============================================================
        # Cleanup
        # ============================================================

        del (
            batch_obs,
            batch_acts,
            batch_log_probs,
            batch_rtgs,
            V_old,
            V,
            A_k,
            value_targets,
            curr_log_probs,
            entropy,
        )

    # ================================================================
    # Epoch-level losses
    # ================================================================

    if conf.logs != "disabled":

        conf.writer.add_scalar(
            "Epoch_Loss/critic_loss",
            epoch_critic_loss
            / len(data_loader),
            epoch,
        )

        conf.writer.add_scalar(
            "Epoch_Loss/actor_loss",
            epoch_actor_loss
            / len(data_loader),
            epoch,
        )


@torch.no_grad()
def evaluate_foundation(
    model,
    fglobal,
    data_loader,
    header,
    device,
    epoch,
    conf,
):
    model.eval()
    fglobal.eval()

    metric_logger = MetricLogger(delimiter=" ")

    for data in metric_logger.log_every(
        data_loader,
        100,
        header,
    ):

        # ============================================================
        # Data
        # ============================================================

        hr_features = data["hr"][0].to(
            device,
            dtype=torch.float32,
        )

        state = data["lr"].to(
            device,
            dtype=torch.float32,
        ).clone()

        # ============================================================
        # Global rank reward
        # ============================================================

        global_rank_rewards, _ = calculate_global_rank_rewards(
            combined_hr_features=hr_features,
            method=conf.global_reward_method,
            k=conf.global_reward_k,
        )

        # ============================================================
        # Environment
        # ============================================================

        env = WSICosineObservationEnv(
            lr_features=state.clone(),
            hr_features=hr_features,
            conf=conf,
        )

        env.reset()

        # ============================================================
        # Episode variables
        # ============================================================

        visited_patch_ids = []
        current_best_coverage = None

        total_reward = 0.0

        ssl_values = []
        diversity_values = []
        coverage_values = []
        global_values = []

        initial_ssl = ssl_state_loss(
            state,
            hr_features,
        ).item()

        done = False

        # ============================================================
        # Rollout
        #
        # IMPORTANT:
        # Just like training/SASHA, the ENVIRONMENT determines when
        # the episode ends.
        # ============================================================

        while not done:

            # --------------------------------------------------------
            # Deterministic action for validation/test
            # --------------------------------------------------------

            action, _, _ = model.get_action(
                state,
                visited_patch_ids,
                is_eval=True,
            )

            action_idx = int(
                action.item()
            )

            # --------------------------------------------------------
            # SASHA-style environment can become "fully covered"
            # before the time horizon ends.
            #
            # After this happens env.step() is effectively a no-op:
            # state does not change, it only advances time.
            #
            # Therefore those actor choices should receive 0 reward.
            # --------------------------------------------------------

            state_was_fully_covered = (
                env.visited_all_patches
            )

            # --------------------------------------------------------
            # SSL loss before transition
            # --------------------------------------------------------

            loss_before = ssl_state_loss(
                state,
                hr_features,
            )

            # ========================================================
            # Exploration reward
            # ========================================================

            if not state_was_fully_covered:

                exploration_reward, reward_details = (
                    calculate_exploration_reward(
                        combined_hr_features=hr_features,
                        action_index=action,
                        observed_indices=visited_patch_ids,
                        global_rank_rewards=global_rank_rewards,
                        current_best_coverage=current_best_coverage,
                        lambda_global=conf.lambda_global,
                        lambda_diversity=conf.lambda_diversity,
                        lambda_coverage=conf.lambda_coverage,
                        reward_scale=conf.reward_scale,
                    )
                )

                current_best_coverage = (
                    reward_details[
                        "updated_best_coverage"
                    ]
                )

                global_reward = (
                    reward_details[
                        "global_reward"
                    ]
                )

                diversity_reward = (
                    reward_details[
                        "diversity_reward"
                    ]
                )

                coverage_reward = (
                    reward_details[
                        "coverage_reward"
                    ]
                )

            else:

                exploration_reward = torch.tensor(
                    0.0,
                    dtype=torch.float32,
                    device=device,
                )

                global_reward = torch.tensor(
                    0.0,
                    dtype=torch.float32,
                    device=device,
                )

                diversity_reward = torch.tensor(
                    0.0,
                    dtype=torch.float32,
                    device=device,
                )

                coverage_reward = torch.tensor(
                    0.0,
                    dtype=torch.float32,
                    device=device,
                )

            # ========================================================
            # Environment transition
            # ========================================================

            new_state, _, done = env.step(
                action=action,
                state_update_net=fglobal,
                device=device,
            )

            # --------------------------------------------------------
            # SSL loss after transition
            # --------------------------------------------------------

            loss_after = ssl_state_loss(
                new_state,
                hr_features,
            )

            # ========================================================
            # SSL reward
            # ========================================================

            if not state_was_fully_covered:

                ssl_reward = (
                    conf.reward_scale
                    * conf.lambda_ssl
                    * (
                        loss_before
                        - loss_after
                    )
                )

            else:

                ssl_reward = torch.tensor(
                    0.0,
                    dtype=torch.float32,
                    device=device,
                )

            # ========================================================
            # Total foundation reward
            # ========================================================

            reward = (
                exploration_reward
                + ssl_reward
            )

            total_reward += float(
                reward.item()
            )

            # ========================================================
            # Diagnostics
            # ========================================================

            ssl_values.append(
                float(
                    ssl_reward.item()
                )
            )

            global_values.append(
                float(
                    global_reward.item()
                )
            )

            diversity_values.append(
                float(
                    diversity_reward.item()
                )
            )

            coverage_values.append(
                float(
                    coverage_reward.item()
                )
            )

            # --------------------------------------------------------
            # Keep same actor masking logic as training/SASHA:
            # every actor-selected action goes into this list.
            # --------------------------------------------------------

            visited_patch_ids.append(
                action_idx
            )

            state = new_state

        # ============================================================
        # Final diagnostics
        # ============================================================

        final_ssl = ssl_state_loss(
            state,
            hr_features,
        ).item()

        if current_best_coverage is not None:

            final_coverage = float(
                current_best_coverage
                .mean()
                .item()
            )

        else:

            final_coverage = 0.0

        # ------------------------------------------------------------
        # Mean pairwise cosine between actor-selected patches
        # ------------------------------------------------------------

        if len(
            visited_patch_ids
        ) > 1:

            selected = hr_features[
                torch.as_tensor(
                    visited_patch_ids,
                    device=device,
                    dtype=torch.long,
                )
            ]

            selected = F.normalize(
                selected,
                dim=-1,
            )

            selected_similarity = (
                selected
                @ selected.T
            )

            upper = torch.triu_indices(
                selected.shape[0],
                selected.shape[0],
                offset=1,
                device=device,
            )

            selected_pairwise_cosine = float(
                selected_similarity[
                    upper[0],
                    upper[1],
                ]
                .mean()
                .item()
            )

        else:

            selected_pairwise_cosine = 0.0

        # ============================================================
        # Safe diagnostic means
        # ============================================================

        mean_global = (
            float(
                np.mean(
                    global_values
                )
            )
            if len(global_values) > 0
            else 0.0
        )

        mean_diversity = (
            float(
                np.mean(
                    diversity_values
                )
            )
            if len(diversity_values) > 0
            else 0.0
        )

        mean_coverage = (
            float(
                np.mean(
                    coverage_values
                )
            )
            if len(coverage_values) > 0
            else 0.0
        )

        mean_ssl = (
            float(
                np.mean(
                    ssl_values
                )
            )
            if len(ssl_values) > 0
            else 0.0
        )

        # ============================================================
        # Metric logger
        # ============================================================

        metric_logger.update(
            reward=total_reward
        )

        metric_logger.update(
            global_reward=mean_global
        )

        metric_logger.update(
            diversity_reward=mean_diversity
        )

        metric_logger.update(
            coverage_reward=mean_coverage
        )

        metric_logger.update(
            ssl_reward=mean_ssl
        )

        metric_logger.update(
            final_coverage=final_coverage
        )

        metric_logger.update(
            selected_pairwise_cosine=
            selected_pairwise_cosine
        )

        metric_logger.update(
            initial_ssl_loss=initial_ssl
        )

        metric_logger.update(
            final_ssl_loss=final_ssl
        )

    # ================================================================
    # Print
    # ================================================================

    print(
        f"* {header} "
        f"reward {metric_logger.reward.global_avg:.4f} "
        f"global {metric_logger.global_reward.global_avg:.4f} "
        f"diversity {metric_logger.diversity_reward.global_avg:.4f} "
        f"coverage_gain {metric_logger.coverage_reward.global_avg:.4f} "
        f"ssl {metric_logger.ssl_reward.global_avg:.4f} "
        f"final_coverage {metric_logger.final_coverage.global_avg:.4f} "
        f"selected_cos {metric_logger.selected_pairwise_cosine.global_avg:.4f} "
        f"initial_ssl_diag {metric_logger.initial_ssl_loss.global_avg:.4f} "
        f"final_ssl_diag {metric_logger.final_ssl_loss.global_avg:.4f}"
    )

    # ================================================================
    # WandB
    # ================================================================

    log_payload = {
        f"{header.lower()}/reward":
            metric_logger.reward.global_avg,

        f"{header.lower()}/global_reward":
            metric_logger.global_reward.global_avg,

        f"{header.lower()}/diversity_reward":
            metric_logger.diversity_reward.global_avg,

        f"{header.lower()}/coverage_reward":
            metric_logger.coverage_reward.global_avg,

        f"{header.lower()}/ssl_reward":
            metric_logger.ssl_reward.global_avg,

        f"{header.lower()}/final_coverage":
            metric_logger.final_coverage.global_avg,

        f"{header.lower()}/selected_pairwise_cosine":
            metric_logger.selected_pairwise_cosine.global_avg,

        f"{header.lower()}/initial_ssl_loss_diagnostic":
            metric_logger.initial_ssl_loss.global_avg,

        f"{header.lower()}/final_ssl_loss_diagnostic":
            metric_logger.final_ssl_loss.global_avg,

        "epoch":
            epoch,
    }

    wandb.log(
        log_payload
    )

    # ================================================================
    # TensorBoard
    # ================================================================

    if conf.logs != "disabled":

        conf.writer.add_scalar(
            f"{header}/reward",
            metric_logger.reward.global_avg,
            epoch,
        )

        conf.writer.add_scalar(
            f"{header}/global_reward",
            metric_logger.global_reward.global_avg,
            epoch,
        )

        conf.writer.add_scalar(
            f"{header}/diversity_reward",
            metric_logger.diversity_reward.global_avg,
            epoch,
        )

        conf.writer.add_scalar(
            f"{header}/coverage_reward",
            metric_logger.coverage_reward.global_avg,
            epoch,
        )

        conf.writer.add_scalar(
            f"{header}/ssl_reward",
            metric_logger.ssl_reward.global_avg,
            epoch,
        )

        conf.writer.add_scalar(
            f"{header}/final_coverage",
            metric_logger.final_coverage.global_avg,
            epoch,
        )

        conf.writer.add_scalar(
            f"{header}/selected_pairwise_cosine",
            metric_logger.selected_pairwise_cosine.global_avg,
            epoch,
        )

        conf.writer.add_scalar(
            f"{header}/initial_ssl_loss_diagnostic",
            metric_logger.initial_ssl_loss.global_avg,
            epoch,
        )

        conf.writer.add_scalar(
            f"{header}/final_ssl_loss_diagnostic",
            metric_logger.final_ssl_loss.global_avg,
            epoch,
        )

    # ================================================================
    # Return values used for checkpoint selection
    # ================================================================

    return (
        metric_logger.final_ssl_loss.global_avg,
        metric_logger.reward.global_avg,
    )

def train_one_epoch(model, fglobal, classifier, data_loader, actor_optimizer, critic_optimizer, device, epoch, conf):
    model.train()
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter('actor_lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('critic_lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 100

    epoch_actor_loss = 0
    epoch_critic_loss = 0
    for data_it, data in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        # torch.cuda.empty_cache()
        hr_features = data['hr'][0].to(device, dtype=torch.float32)
        state = data['lr'].to(device, dtype=torch.float32).clone()
        label = data['label'].to(device)
        if conf.fglobal == 'attn':
            env = WSIObservationEnv(lr_features=state, hr_features=hr_features, label=label, conf=conf)
        else:
            env = WSICosineObservationEnv(lr_features=state, hr_features=hr_features, label=label, conf=conf)

        # adjust_learning_rate(actor_optimizer, epoch + data_it / len(data_loader), conf)
        # adjust_learning_rate(critic_optimizer, epoch + data_it / len(data_loader), conf)

        # collecting data as PPO is a onpolicy algorithm
        batch_obs = []
        batch_acts = []
        batch_log_probs = []
        batch_rews = []
        batch_lens = []
        dones = []
        N = state.shape[1]
        T = min(N, int(conf.frac_visit * N))
        with torch.no_grad():
            for i in range(conf.num_envs):
                ep_rews = []
                visited_patch_id = []
                env.reset()
                done = False
                ep_t = 0
                while not done:
                    # track observations in this batch
                    batch_obs.append(state)
                    action, log_prob, entropy = model.get_action(state, visited_patch_id, is_eval = False)
                    state, reward, done = env.step(action=action, state_update_net=fglobal, classifier_net=classifier,
                                                   device=device)
                    ep_rews.append(reward)
                    batch_acts.append(action.item())
                    batch_log_probs.append(log_prob.item())
                    visited_patch_id.append(action.item())
                    dones.append(done)
                    ep_t += 1
                state = data['lr'].to(device, dtype=torch.float32).clone()
                batch_lens.append(ep_t + 1)
                batch_rews.append(ep_rews)

            batch_obs = torch.stack(batch_obs).squeeze(1)
            batch_acts = torch.tensor(batch_acts, dtype=torch.float, device=conf.device)
            batch_log_probs = torch.tensor(batch_log_probs, dtype=torch.float, device=conf.device)
            batch_rtgs = compute_rtgs(batch_rews, conf).to(conf.device)
            V_old, _, _ = model.evaluate(batch_obs, batch_acts)

            if conf.use_gae:
                V_numpy = V_old.view(conf.num_envs, -1).cpu().numpy()
                dones_numpy = np.array(dones).reshape(conf.num_envs, -1)
                rews_numpy = np.array(batch_rews).reshape(conf.num_envs, -1)
                
                A_k = gae(rewards=rews_numpy, values=V_numpy, episode_ends=dones_numpy, gamma=conf.gamma, lam=conf.lam)
                A_k = torch.tensor(A_k, dtype=torch.float, device=conf.device).flatten()
                
                # Set the critic target to the GAE returns
                batch_rtgs = A_k + V_old.detach().flatten()

            else:
                A_k = batch_rtgs - V_old.detach().flatten()

            A_k = (A_k - A_k.mean()) / (A_k.std() + 1e-8)

        # updating parameters for n epochs

        for _ in range(conf.num_epochs_on_single_roll_out):
            # torch.cuda.empty_cache()
            V, curr_log_probs, entropy = model.evaluate(batch_obs, batch_acts)
            ratios = torch.exp(curr_log_probs - batch_log_probs)

            # calculate surrogate loss
            surr1 = ratios * A_k
            surr2 = torch.clamp(ratios, 1 - conf.clip, 1 + conf.clip) * A_k
            actor_loss = (-torch.min(surr1, surr2)).mean()

            # Critic loss
            critic_loss = nn.MSELoss()(V.flatten(), batch_rtgs)
            entropy_loss = entropy.mean()

            if conf.use_entropy_loss:
                actor_loss -= conf.entropy_coef * entropy_loss

            # updating actor
            actor_optimizer.zero_grad()
            actor_loss.backward()
            nn.utils.clip_grad_norm_(model.actor.parameters(), conf.max_grad_norm)
            actor_optimizer.step()

            # updating critic
            critic_optimizer.zero_grad()
            critic_loss.backward()
            nn.utils.clip_grad_norm_(model.critic.parameters(), conf.max_grad_norm)
            critic_optimizer.step()

            epoch_actor_loss += actor_loss.item()
            epoch_critic_loss += critic_loss.item()

            # Clear CUDA cache
            # torch.cuda.empty_cache()
            # gc.collect()

        metric_logger.update(actor_lr=actor_optimizer.param_groups[0]['lr'])
        metric_logger.update(critic_lr=critic_optimizer.param_groups[0]['lr'])
        metric_logger.update(actor_loss=actor_loss.item())
        metric_logger.update(critic_loss=critic_loss.item())

        if conf.logs != 'disabled':
            conf.writer.add_scalar("Loss/critic_loss", critic_loss.item(), data_it + (epoch * len(data_loader)))
            conf.writer.add_scalar("Loss/actor_loss", actor_loss.item(), data_it + (epoch * len(data_loader)))

        # del batch_obs, batch_acts, batch_log_probs, batch_rtgs, V, curr_log_probs, ratios, surr1, surr2, A_k

        # # Clear CUDA cache
        # torch.cuda.empty_cache()
        # gc.collect()

    if conf.logs != 'disabled':
        conf.writer.add_scalar("Epoch_Loss/critic_loss", epoch_critic_loss / len(data_loader), epoch)
        conf.writer.add_scalar("Epoch_Loss/actor_loss", epoch_actor_loss / len(data_loader), epoch)


@torch.no_grad()
def evaluate_policy(model, fglobal, classifier, data_loader, header, device, epoch, conf):
    model.eval()

    y_pred = []
    y_true = []
    metric_logger = MetricLogger(delimiter=" ")
    final_reward = 0

    for data in metric_logger.log_every(data_loader, 100, header):
        hr_features = data['hr'][0].to(device, dtype=torch.float32)
        state = data['lr'].to(device, dtype=torch.float32)
        label = data['label'].to(device)

        if conf.fglobal == 'attn':
            env = WSIObservationEnv(lr_features=state, hr_features=hr_features, label=label, conf=conf)
        else:
            env = WSICosineObservationEnv(lr_features=state, hr_features=hr_features, label=label, conf=conf)

        N = state.shape[1]
        visited_patch_id = []
        done = False
        while not done:
            action, log_prob, entropy = model.get_action(state, visited_patch_id, is_eval = True)
            new_state, reward, done = env.step(action=action, state_update_net=fglobal, classifier_net=classifier,
                                               device=device)
            state = new_state

        final_reward += reward
        loss = -1 * reward
        slide_preds, attn = classifier.classify(state)
        pred = torch.softmax(slide_preds, dim=-1)
        acc1 = accuracy(pred, label, topk=(1,))[0]
        metric_logger.update(loss=loss)
        metric_logger.meters['acc1'].update(acc1.item(), n=label.shape[0])

        y_pred.append(pred)
        y_true.append(label)

    y_pred = torch.cat(y_pred, dim=0)
    y_true = torch.cat(y_true, dim=0)
    y_pred_labels = torch.argmax(y_pred, dim=-1)
    AUROC_metric = torchmetrics.AUROC(task='binary').to(device)
    AUROC_metric(y_pred[:, 1], y_true)
    auroc = AUROC_metric.compute().item()
    F1_metric = torchmetrics.F1Score(task='binary').to(device)
    F1_metric(y_pred_labels, y_true)
    f1_score = F1_metric.compute().item()

    Precision_metric = torchmetrics.Precision(task='binary').to(device)
    Precision_metric(y_pred_labels, y_true)
    precision = Precision_metric.compute().item()

    Recall_metric = torchmetrics.Recall(task='binary').to(device)
    Recall_metric(y_pred_labels, y_true)
    recall = Recall_metric.compute().item()

    y_pred_np = y_pred_labels.cpu().numpy()
    y_true_np = y_true.cpu().numpy()
    balanced_acc = balanced_accuracy_score(y_true_np, y_pred_np)

    print('* Acc@1 {top1.global_avg:.3f} loss {losses.global_avg:.3f} auroc {AUROC:.3f} f1_score {F1:.3f}'
          .format(top1=metric_logger.acc1, losses=metric_logger.loss, AUROC=auroc, F1=f1_score))

    if conf.logs != 'disabled':
        conf.writer.add_scalar(f"{header}/accuracy", metric_logger.acc1.global_avg, epoch)
        conf.writer.add_scalar(f"{header}/auroc", auroc, epoch)
        conf.writer.add_scalar(f"{header}/f1", f1_score, epoch)
        conf.writer.add_scalar(f"{header}/loss", metric_logger.loss.global_avg, epoch)


    return auroc, metric_logger.acc1.global_avg, f1_score, metric_logger.loss.global_avg, balanced_acc, precision, recall


if __name__ == '__main__':
    random_seed = 1130
    main(random_seed=random_seed)