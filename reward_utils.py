from __future__ import annotations

from typing import Literal, Sequence

import torch
import torch.nn.functional as F


def _normalize_features(features: torch.Tensor) -> torch.Tensor:
    """
    Normalize a feature matrix for cosine-similarity calculations.

    Parameters
    ----------
    features:
        Tensor of shape [N, D].

    Returns
    -------
    Tensor of shape [N, D] with unit-length rows.
    """
    if features.ndim != 2:
        raise ValueError(
            f"Expected features with shape [N, D], got {tuple(features.shape)}."
        )

    if features.shape[0] == 0:
        raise ValueError("The feature matrix contains zero patches.")

    if not torch.isfinite(features).all():
        raise ValueError("Features contain NaN or Inf values.")

    return F.normalize(
        features.float(),
        p=2,
        dim=-1,
        eps=1e-8,
    )


def _scaled_cosine_similarity(
    normalized_a: torch.Tensor,
    normalized_b: torch.Tensor,
) -> torch.Tensor:
    """
    Compute cosine similarity and map it from [-1, 1] to [0, 1].

    Inputs must already be L2-normalized.
    """
    cosine = normalized_a @ normalized_b.transpose(-1, -2)

    return ((cosine + 1.0) / 2.0).clamp(
        min=0.0,
        max=1.0,
    )

@torch.no_grad()
def calculate_global_rank_rewards(
    combined_hr_features: torch.Tensor,
    method: Literal[
        "knn_mean",
        "global_mean",
        "max",
    ] = "knn_mean",
    k: int = 20,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Calculate a global uniqueness-rank reward for every patch.

    Parameters
    ----------
    combined_hr_features:
        Combined HR representations with shape [N, D].

    method:
        "knn_mean":
            Uniqueness is one minus the mean similarity to the k nearest
            other patches. Recommended.

        "global_mean":
            Uniqueness is one minus the mean similarity to every other patch.

        "max":
            Uniqueness is one minus the maximum similarity to another patch.

    k:
        Number of nearest neighbors used with method="knn_mean".
        Self-similarity is excluded.

    Returns
    -------
    rank_rewards:
        Tensor of shape [N], with rewards in [0, 1].

        0 means the least globally unique patch.
        1 means the most globally unique patch.

    raw_uniqueness:
        Tensor of shape [N] containing the uniqueness score before ranking.
    """
    features = _normalize_features(
        combined_hr_features
    )

    num_patches = features.shape[0]

    if num_patches == 1:
        return (
            torch.ones(
                1,
                device=features.device,
                dtype=features.dtype,
            ),
            torch.ones(
                1,
                device=features.device,
                dtype=features.dtype,
            ),
        )

    similarity_matrix = _scaled_cosine_similarity(
        features,
        features,
    )

    # Exclude each patch's similarity with itself.
    similarity_without_self = similarity_matrix.clone()

    similarity_without_self.fill_diagonal_(
        float("-inf")
    )

    if method == "knn_mean":
        effective_k = min(
            max(int(k), 1),
            num_patches - 1,
        )

        nearest_similarities = torch.topk(
            similarity_without_self,
            k=effective_k,
            dim=1,
            largest=True,
        ).values

        average_similarity = (
            nearest_similarities.mean(dim=1)
        )

    elif method == "global_mean":
        # Restore a finite value before summing and explicitly exclude self.
        similarity_without_self = similarity_matrix.clone()
        similarity_without_self.fill_diagonal_(0.0)

        average_similarity = (
            similarity_without_self.sum(dim=1)
            / float(num_patches - 1)
        )

    elif method == "max":
        average_similarity = (
            similarity_without_self.max(dim=1).values
        )

    else:
        raise ValueError(
            f"Unsupported global uniqueness method: {method}"
        )

    raw_uniqueness = (
        1.0 - average_similarity
    ).clamp(
        min=0.0,
        max=1.0,
    )

    # Percentile rank:
    # least unique -> 0
    # most unique  -> 1
    sorted_indices = torch.argsort(
        raw_uniqueness,
        descending=False,
    )

    integer_ranks = torch.empty(
        num_patches,
        device=features.device,
        dtype=torch.long,
    )

    integer_ranks[sorted_indices] = torch.arange(
        num_patches,
        device=features.device,
    )

    rank_rewards = (
        integer_ranks.float()
        / float(num_patches - 1)
    )

    return rank_rewards, raw_uniqueness


@torch.no_grad()
def calculate_observed_diversity_reward(
    combined_hr_features: torch.Tensor,
    action_index: int | torch.Tensor,
    observed_indices: Sequence[int] | torch.Tensor,
    reduction: Literal["max", "mean"] = "max",
    first_action_reward: float = 0.0,
) -> torch.Tensor:
    """
    Reward a newly selected patch for being different from previously
    observed patches.

    Parameters
    ----------
    combined_hr_features:
        Combined HR representations with shape [N, D].

    action_index:
        Index of the newly selected patch.

    observed_indices:
        Indices selected before the current action.

        Do not include action_index in this list when calling the function.

    reduction:
        "max":
            Penalizes the candidate based on its most similar observed patch.
            Recommended because it detects near-duplicates.

        "mean":
            Uses average similarity to the observed set.

    first_action_reward:
        Reward returned when no patches have previously been observed.

    Returns
    -------
    Scalar tensor, normally in [0, 1].

    Higher means the selected patch is more different from previously
    observed patches.
    """
    features = _normalize_features(
        combined_hr_features
    )

    num_patches = features.shape[0]

    if isinstance(action_index, torch.Tensor):
        action_index = int(action_index.item())
    else:
        action_index = int(action_index)

    if not 0 <= action_index < num_patches:
        raise IndexError(
            f"action_index={action_index} is outside [0, {num_patches - 1}]."
        )

    if isinstance(observed_indices, torch.Tensor):
        observed = observed_indices.to(
            device=features.device,
            dtype=torch.long,
        ).flatten()
    else:
        observed = torch.tensor(
            list(observed_indices),
            device=features.device,
            dtype=torch.long,
        )

    if observed.numel() == 0:
        return torch.tensor(
            first_action_reward,
            device=features.device,
            dtype=features.dtype,
        )

    if (
        observed.min().item() < 0
        or observed.max().item() >= num_patches
    ):
        raise IndexError(
            "observed_indices contains an invalid patch index."
        )

    # Remove duplicates in case the list contains repeated indices.
    observed = torch.unique(observed)

    action_feature = features[
        action_index : action_index + 1
    ]

    observed_features = features[
        observed
    ]

    similarities = _scaled_cosine_similarity(
        action_feature,
        observed_features,
    ).squeeze(0)

    if reduction == "max":
        redundancy = similarities.max()

    elif reduction == "mean":
        redundancy = similarities.mean()

    else:
        raise ValueError(
            f"Unsupported diversity reduction: {reduction}"
        )

    diversity_reward = (
        1.0 - redundancy
    ).clamp(
        min=0.0,
        max=1.0,
    )

    return diversity_reward

@torch.no_grad()
def calculate_coverage_reward(
    combined_hr_features: torch.Tensor,
    action_index: int | torch.Tensor,
    current_best_coverage: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Calculate the marginal facility-location coverage gained by selecting
    one patch.

    Parameters
    ----------
    combined_hr_features:
        Combined HR representations with shape [N, D].

    action_index:
        Index of the newly selected patch.

    current_best_coverage:
        Tensor of shape [N].

        Entry i stores the highest scaled cosine similarity between patch i
        and any patch selected before this action.

        For the first action, pass None.

    Returns
    -------
    coverage_reward:
        Scalar marginal coverage gain.

    updated_best_coverage:
        Tensor of shape [N]. Store this in the environment for the next step.

    coverage_before:
        Mean WSI coverage before selecting the action.

    coverage_after:
        Mean WSI coverage after selecting the action.
    """
    features = _normalize_features(
        combined_hr_features
    )

    num_patches = features.shape[0]

    if isinstance(action_index, torch.Tensor):
        action_index = int(action_index.item())
    else:
        action_index = int(action_index)

    if not 0 <= action_index < num_patches:
        raise IndexError(
            f"action_index={action_index} is outside [0, {num_patches - 1}]."
        )

    if current_best_coverage is None:
        # Scaled cosine similarity lies in [0, 1], so zero means
        # no part of the WSI has yet been covered.
        current_best_coverage = torch.zeros(
            num_patches,
            device=features.device,
            dtype=features.dtype,
        )

    else:
        current_best_coverage = current_best_coverage.to(
            device=features.device,
            dtype=features.dtype,
        ).flatten()

        if current_best_coverage.shape != (num_patches,):
            raise ValueError(
                "current_best_coverage must have shape "
                f"[{num_patches}], got {tuple(current_best_coverage.shape)}."
            )

        if not torch.isfinite(
            current_best_coverage
        ).all():
            raise ValueError(
                "current_best_coverage contains NaN or Inf."
            )

    candidate_feature = features[
        action_index : action_index + 1
    ]

    # Similarity of every WSI patch to the newly selected representative.
    candidate_coverage = _scaled_cosine_similarity(
        features,
        candidate_feature,
    ).squeeze(1)

    updated_best_coverage = torch.maximum(
        current_best_coverage,
        candidate_coverage,
    )

    coverage_before = (
        current_best_coverage.mean()
    )

    coverage_after = (
        updated_best_coverage.mean()
    )

    coverage_reward = (
        coverage_after - coverage_before
    ).clamp_min(0.0)

    return (
        coverage_reward,
        updated_best_coverage,
        coverage_before,
        coverage_after,
    )

@torch.no_grad()
def calculate_exploration_reward(
    combined_hr_features: torch.Tensor,
    action_index: int | torch.Tensor,
    observed_indices: Sequence[int] | torch.Tensor,
    global_rank_rewards: torch.Tensor,
    current_best_coverage: torch.Tensor | None,
    lambda_global: float = 0.10,
    lambda_diversity: float = 0.30,
    lambda_coverage: float = 0.60,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    Combine global uniqueness, observed-set diversity, and marginal coverage.
    """
    if isinstance(action_index, torch.Tensor):
        action = int(action_index.item())
    else:
        action = int(action_index)

    global_reward = global_rank_rewards[
        action
    ]

    if len(observed_indices) == 0:
        diversity_reward = global_reward
    else:
        diversity_reward = (
            calculate_observed_diversity_reward(
                combined_hr_features=combined_hr_features,
                action_index=action,
                observed_indices=observed_indices,
                reduction="max",
            )
        )

    (
        coverage_reward,
        updated_best_coverage,
        coverage_before,
        coverage_after,
    ) = calculate_coverage_reward(
        combined_hr_features=combined_hr_features,
        action_index=action,
        current_best_coverage=current_best_coverage,
    )

    total_reward = (
        lambda_global * global_reward
        + lambda_diversity * diversity_reward
        + lambda_coverage * coverage_reward
    )

    details = {
        "total_reward": total_reward,
        "global_rank_reward": global_reward,
        "observed_diversity_reward": diversity_reward,
        "coverage_reward": coverage_reward,
        "coverage_before": coverage_before,
        "coverage_after": coverage_after,
        "updated_best_coverage": updated_best_coverage,
    }

    return total_reward, details