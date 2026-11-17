"""
This script initiates the training of the RL Agent component after separately training the HAFED pipeline,
which includes the Feature Aggregator, Classifier and the Tumor Scoring Unit (TSU).

During this phase:
- Pretrained weights for both the Feature Aggregator and TSU are loaded from their respective checkpoints.
- The RL Agent is trained to intelligently select patches based on the above models.


Commands
CAMELYON16
python step6_rl_training.py --config config/camelyon_rl_config.yml --seed 4  --log_dir LOG_DIR

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
from rl_algorithms.ppo import Agent
from rl_algorithms.ppo import TransformerActor as Actor
from rl_algorithms.ppo import TransformerCritic as Critic
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
    nonterminal = 1.0 - episode_ends
    n_envs, horizon = rewards.shape

    advantages = np.zeros_like(rewards, dtype=np.float32)
    gae_step = np.zeros(n_envs, dtype=np.float32)

    for t in reversed(range(horizon)):
        if t == horizon - 1:
            next_value = np.zeros(n_envs, dtype=np.float32)
        else:
            next_value = values[:, t + 1]

        delta = (
            rewards[:, t]
            + gamma * next_value * nonterminal[:, t]
            - values[:, t]
        )

        gae_step = (
            delta
            + gamma * lam * nonterminal[:, t] * gae_step
        )
        advantages[:, t] = gae_step

    return advantages


def gae_(rewards, values, episode_ends, gamma, lam):
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
    for t in reversed(range(T - 1)):
        # First compute delta, which is the one-step TD error
        delta = rewards[:, t] + gamma * values[:, t + 1] * episode_ends[:, t] - values[:, t]
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

def main():
    # getting and config file
    args = get_arguments()


    with open(args.config, 'r') as ymlfile:
        c = yaml.load(ymlfile, Loader=yaml.FullLoader)
        c.update(vars(args))
        conf = Struct(**c)
        conf.log_dir = os.path.join(conf.log_dir, f"cosine_{conf.cosine_threshold}_frac_{conf.frac_visit}_seed_{conf.seed}")

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
    classifier_dict, _, config, _ = load_model(ckpt_path=conf.classifier_ckpt_path, args= args)
    classifier_conf = SimpleNamespace(**config)

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
    actor_optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, actor.parameters()), lr=0.001, weight_decay=conf.wd)
    critic_optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, critic.parameters()), lr=0.001, weight_decay=conf.wd)


    best_state = {
            "epoch": -1,
            "val_final_ssl_loss": float("inf"),
            "val_reward": 0.0,
            "test_final_ssl_loss": float("inf"),
            "test_reward": 0.0,
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

        if val_loss < best_state["val_final_ssl_loss"]:
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
    ).mean() * 10


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

    metric_logger = MetricLogger(delimiter=" ")
    metric_logger.add_meter("actor_lr", SmoothedValue(window_size=1, fmt="{value:.6f}"))
    metric_logger.add_meter("critic_lr", SmoothedValue(window_size=1, fmt="{value:.6f}"))

    header = f"Foundation Epoch: [{epoch}]"
    print_freq = 100

    for data_it, data in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        torch.cuda.empty_cache()

        hr_features = data["hr"][0].to(device, dtype=torch.float32)          # [N, D]
        init_state = data["lr"].to(device, dtype=torch.float32).clone()      # [1, N, D]

        adjust_learning_rate(actor_optimizer, epoch + data_it / len(data_loader), conf)
        adjust_learning_rate(critic_optimizer, epoch + data_it / len(data_loader), conf)

        batch_obs = []
        batch_acts = []
        batch_log_probs = []
        batch_rews = []
        dones = []

        N = init_state.shape[1]
        T = max(1, min(N, int(conf.frac_visit * N)))
        env = WSICosineObservationEnv(
                        lr_features=init_state.clone(),
                        hr_features=hr_features,
                        conf=conf,
                    )

        with torch.no_grad():
            for _ in range(conf.num_envs):
                state = init_state.clone()
                visited_patch_id = []
                ep_rews = []

                for t in range(T):
                    batch_obs.append(state)

                    action, log_prob, entropy = model.get_action(
                        state,
                        visited_patch_id,
                        is_eval=False,
                    )

                    loss_before = ssl_state_loss(state, hr_features)

                    
                    new_state, _, _ = env.step(
                        action=action,
                        state_update_net=fglobal,
                        device=device,
                    )

                    loss_after = ssl_state_loss(new_state, hr_features)

                    reward = (loss_before - loss_after).item()

                    done = t == T - 1

                    ep_rews.append(reward)
                    batch_acts.append(action.item())
                    batch_log_probs.append(log_prob.item())
                    dones.append(done)

                    visited_patch_id.append(action.item())
                    state = new_state

                batch_rews.append(ep_rews)

        batch_obs = torch.stack(batch_obs).squeeze(1)
        batch_acts = torch.tensor(batch_acts, dtype=torch.float32, device=device)
        batch_log_probs = torch.tensor(batch_log_probs, dtype=torch.float32, device=device)
        batch_rtgs = compute_rtgs(batch_rews, conf).to(device)

        V, _, _ = model.evaluate(batch_obs, batch_acts)

        if conf.use_gae:
            V_np = V.view(conf.num_envs, -1).detach().cpu().numpy()
            dones_np = np.array(dones).reshape(conf.num_envs, -1)
            rews_np = np.array(batch_rews).reshape(conf.num_envs, -1)

            A_k = gae(
                rewards=rews_np,
                values=V_np,
                episode_ends=dones_np,
                gamma=conf.gamma,
                lam=conf.lam,
            )

            A_k = torch.tensor(A_k, dtype=torch.float32, device=device).flatten()
        else:
            A_k = batch_rtgs - V.detach()

        A_k = (A_k - A_k.mean()) / (A_k.std() + 1e-8)

        for _ in range(conf.num_epochs_on_single_roll_out):
            V, curr_log_probs, entropy = model.evaluate(batch_obs, batch_acts)

            ratios = torch.exp(curr_log_probs - batch_log_probs)

            surr1 = ratios * A_k
            surr2 = torch.clamp(ratios, 1 - conf.clip, 1 + conf.clip) * A_k

            actor_loss = (-torch.min(surr1, surr2)).mean()
            critic_loss = nn.MSELoss()(V, batch_rtgs)

            if conf.use_entropy_loss:
                actor_loss += conf.entropy_coef * entropy.mean()

            actor_optimizer.zero_grad()
            actor_loss.backward(retain_graph=True)
            nn.utils.clip_grad_norm_(model.actor.parameters(), conf.max_grad_norm)
            actor_optimizer.step()

            critic_optimizer.zero_grad()
            critic_loss.backward()
            nn.utils.clip_grad_norm_(model.critic.parameters(), conf.max_grad_norm)
            critic_optimizer.step()

        mean_ep_reward = float(np.mean([np.sum(r) for r in batch_rews]))
        final_ssl = ssl_state_loss(state, hr_features).item()

        metric_logger.update(actor_lr=actor_optimizer.param_groups[0]["lr"])
        metric_logger.update(critic_lr=critic_optimizer.param_groups[0]["lr"])
        metric_logger.update(actor_loss=actor_loss.item())
        metric_logger.update(critic_loss=critic_loss.item())
        metric_logger.update(reward=mean_ep_reward)
        metric_logger.update(final_ssl_loss=final_ssl)

        if conf.logs != "disabled":
            step = data_it + epoch * len(data_loader)
            conf.writer.add_scalar("Foundation/actor_loss", actor_loss.item(), step)
            conf.writer.add_scalar("Foundation/critic_loss", critic_loss.item(), step)
            conf.writer.add_scalar("Foundation/reward", mean_ep_reward, step)
            conf.writer.add_scalar("Foundation/final_ssl_loss", final_ssl, step)

        del batch_obs, batch_acts, batch_log_probs, batch_rtgs, V, A_k
        torch.cuda.empty_cache()
        gc.collect()


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

    metric_logger = MetricLogger(delimiter=" ")

    for data in metric_logger.log_every(data_loader, 100, header):
        hr_features = data["hr"][0].to(device, dtype=torch.float32)
        state = data["lr"].to(device, dtype=torch.float32).clone()

        N = state.shape[1]
        T = max(1, min(N, int(conf.frac_visit * N)))

        visited_patch_id = []
        total_reward = 0.0

        initial_loss = ssl_state_loss(state, hr_features).item()

        env = WSICosineObservationEnv(
                lr_features=state,
                hr_features=hr_features,
                conf=conf,
            )

        for t in range(T):
            action, log_prob, entropy = model.get_action(
                state,
                visited_patch_id,
                is_eval=True,
            )

            loss_before = ssl_state_loss(state, hr_features)

            new_state, _, _ = env.step(
                action=action,
                state_update_net=fglobal,
                device=device,
            )

            loss_after = ssl_state_loss(new_state, hr_features)

            reward = (loss_before - loss_after).item()
            total_reward += reward

            visited_patch_id.append(action.item())
            state = new_state

        final_loss = ssl_state_loss(state, hr_features).item()
        improvement = initial_loss - final_loss

        metric_logger.update(reward=total_reward)
        metric_logger.update(initial_ssl_loss=initial_loss)
        metric_logger.update(final_ssl_loss=final_loss)
        metric_logger.update(improvement=improvement)

    print(
        f"* {header} "
        f"reward {metric_logger.reward.global_avg:.4f} "
        f"initial_ssl_loss {metric_logger.initial_ssl_loss.global_avg:.4f} "
        f"final_ssl_loss {metric_logger.final_ssl_loss.global_avg:.4f} "
        f"improvement {metric_logger.improvement.global_avg:.4f}"
    )

    if conf.logs != "disabled":
        conf.writer.add_scalar(f"{header}/reward", metric_logger.reward.global_avg, epoch)
        conf.writer.add_scalar(f"{header}/initial_ssl_loss", metric_logger.initial_ssl_loss.global_avg, epoch)
        conf.writer.add_scalar(f"{header}/final_ssl_loss", metric_logger.final_ssl_loss.global_avg, epoch)
        conf.writer.add_scalar(f"{header}/improvement", metric_logger.improvement.global_avg, epoch)

    return metric_logger.final_ssl_loss.global_avg, metric_logger.reward.global_avg

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
        torch.cuda.empty_cache()
        hr_features = data['hr'][0].to(device, dtype=torch.float32)
        state = data['lr'].to(device, dtype=torch.float32).clone()
        label = data['label'].to(device)
        if conf.fglobal == 'attn':
            env = WSIObservationEnv(lr_features=state, hr_features=hr_features, label=label, conf=conf)
        else:
            env = WSICosineObservationEnv(lr_features=state, hr_features=hr_features, label=label, conf=conf)

        adjust_learning_rate(actor_optimizer, epoch + data_it / len(data_loader), conf)
        adjust_learning_rate(critic_optimizer, epoch + data_it / len(data_loader), conf)

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
            V, _, _ = model.evaluate(batch_obs, batch_acts)

            if conf.use_gae:
                V = V.view(conf.num_envs, -1).cpu().numpy()
                dones = np.array(dones).reshape(conf.num_envs, -1)
                batch_rews = np.array(batch_rews).reshape(conf.num_envs, -1)
                A_k = gae(rewards=batch_rews, values=V, episode_ends=dones, gamma=conf.gamma, lam=conf.lam)
                A_k = torch.tensor(A_k, device=conf.device).flatten()

            else:
                A_k = batch_rtgs - V.detach()

            A_k = (A_k - A_k.mean()) / (A_k.std() + 1e-8)

        # updating parameters for n epochs

        for _ in range(conf.num_epochs_on_single_roll_out):
            torch.cuda.empty_cache()
            V, curr_log_probs, entropy = model.evaluate(batch_obs, batch_acts)
            ratios = torch.exp(curr_log_probs - batch_log_probs)

            # calculate surrogate loss
            surr1 = ratios * A_k
            surr2 = torch.clamp(ratios, 1 - conf.clip, 1 + conf.clip) * A_k
            actor_loss = (-torch.min(surr1, surr2)).mean()
            critic_loss = nn.MSELoss()(V, batch_rtgs)
            entropy_loss = entropy.mean()
            if conf.use_entropy_loss:
                actor_loss += conf.entropy_coef * entropy_loss

            # updating actor
            actor_optimizer.zero_grad()
            actor_loss.backward(retain_graph=True)
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
            torch.cuda.empty_cache()
            gc.collect()

        metric_logger.update(actor_lr=actor_optimizer.param_groups[0]['lr'])
        metric_logger.update(critic_lr=critic_optimizer.param_groups[0]['lr'])
        metric_logger.update(actor_loss=actor_loss.item())
        metric_logger.update(critic_loss=critic_loss.item())

        if conf.logs != 'disabled':
            conf.writer.add_scalar("Loss/critic_loss", critic_loss.item(), data_it + (epoch * len(data_loader)))
            conf.writer.add_scalar("Loss/actor_loss", actor_loss.item(), data_it + (epoch * len(data_loader)))

        del batch_obs, batch_acts, batch_log_probs, batch_rtgs, V, curr_log_probs, ratios, surr1, surr2, A_k

        # Clear CUDA cache
        torch.cuda.empty_cache()
        gc.collect()

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
    main()