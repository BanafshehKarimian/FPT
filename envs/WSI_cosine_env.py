import math

import torch
import torch.nn.functional as F


class WSICosineObservationEnv:
    def __init__(
        self,
        lr_features,
        hr_features,
        conf,
    ):
        self.high_cosine_indices = None

        self.conf = conf

        self.features = lr_features.clone()
        self.current_state = (
            self.features.clone()
        )

        self.hr_features = hr_features

        self.B, self.N, self.d = (
            self.current_state.shape
        )

        self.current_time_step = 0

        self.max_time_steps = math.ceil(
            self.N * conf.frac_visit
        )

        self.visited_patches = torch.zeros(
            size=(self.N,),
            dtype=torch.bool,
            device=conf.device,
        )

        self.visited_patch_idx = []

        self.mask = torch.ones(
            (self.N,),
            device=conf.device,
        )

        self.cosine_threshold = (
            conf.cosine_threshold
        )

        self.visited_all_patches = False


    def reset(self):
        self.current_state = (
            self.features.clone()
        )

        self.current_time_step = 0

        self.visited_patches = torch.zeros(
            size=(self.N,),
            dtype=torch.bool,
            device=self.conf.device,
        )

        self.visited_patch_idx = []

        self.mask = torch.ones(
            (self.N,),
            device=self.conf.device,
        )

        self.high_cosine_indices = None

        self.visited_all_patches = False


    def ssl_state_loss(
        self,
        state=None,
    ):
        if state is None:
            state = self.current_state

        if state.dim() == 3:
            state = state.squeeze(0)

        hr = self.hr_features

        if hr.dim() == 3:
            hr = hr.squeeze(0)

        mean_cosine = F.cosine_similarity(
            F.normalize(
                state,
                dim=-1,
            ),
            F.normalize(
                hr,
                dim=-1,
            ),
            dim=-1,
        ).mean()

        return 10.0 * (
            1.0 - mean_cosine
        )


    @torch.no_grad()
    def step(
        self,
        action,
        state_update_net,
        device,
    ):

        if isinstance(
            action,
            torch.Tensor,
        ):
            action = int(
                action.item()
            )
        else:
            action = int(action)

        # ====================================================
        # State cannot change anymore once all regions
        # are marked covered/visited.
        # ====================================================

        if self.visited_all_patches:

            self.current_time_step += 1

            reward = 0.0

            done = (
                self.current_time_step
                >= self.max_time_steps
            )

            if done:
                final_state = (
                    self.current_state.clone()
                )

                self.reset()

                return (
                    final_state,
                    reward,
                    done,
                )

            return (
                self.current_state.clone(),
                reward,
                done,
            )

        # ====================================================
        # Normal transition
        # ====================================================

        if (
            self.current_time_step
            < self.max_time_steps
        ):

            # -----------------------------------------------
            # SSL state before the action
            # -----------------------------------------------

            loss_before = (
                self.ssl_state_loss(
                    self.current_state
                )
            )

            self.current_time_step += 1

            # Explicitly selected action is visited.
            self.visited_patches[
                action
            ] = True

            self.visited_patch_idx.append(
                action
            )

            # -----------------------------------------------
            # Regions still eligible for propagation
            # -----------------------------------------------

            valid_indices = (
                self.mask == 1
            ).nonzero(
                as_tuple=False
            )[:, 0]

            # -----------------------------------------------
            # Similarity based on current state
            # -----------------------------------------------

            cosine_vector = (
                torch.cosine_similarity(
                    self.current_state[0],
                    self.current_state[
                        0,
                        action,
                    ].unsqueeze(0),
                    dim=-1,
                )
            )

            high_cosine_indices = (
                torch.abs(
                    cosine_vector
                )
                >= self.cosine_threshold
            ).nonzero(
                as_tuple=False
            )[:, 0]

            # =================================================
            # IMPORTANT:
            # Match the NEW SASHA environment.
            #
            # Similar patches are now considered covered and
            # cannot be updated through propagation again.
            # =================================================

            self.mask[
                high_cosine_indices
            ] = 0

            self.visited_patches[
                high_cosine_indices
            ] = True

            # Remove explicitly selected action.
            high_cosine_indices = (
                high_cosine_indices[
                    high_cosine_indices
                    != action
                ]
            )

            # Only patches that were valid BEFORE this
            # transition should be propagated to.
            if (
                high_cosine_indices.numel()
                > 0
            ):

                valid_set = set(
                    valid_indices.tolist()
                )

                high_cosine_indices = (
                    torch.tensor(
                        [
                            idx
                            for idx
                            in high_cosine_indices.tolist()
                            if idx
                            in valid_set
                        ],
                        device=device,
                        dtype=torch.long,
                    )
                )

            else:

                high_cosine_indices = (
                    torch.empty(
                        0,
                        device=device,
                        dtype=torch.long,
                    )
                )

            self.high_cosine_indices = (
                high_cosine_indices
            )

            # -----------------------------------------------
            # True HR representation for selected region
            # -----------------------------------------------

            v_at = self.hr_features[
                action
            ].to(
                device
            )

            self.current_state = (
                self.current_state.clone()
            )

            # -----------------------------------------------
            # No similar unobserved patches
            # -----------------------------------------------

            if (
                high_cosine_indices.shape[0]
                == 0
            ):

                self.current_state[
                    0,
                    action,
                    :
                ] = v_at

            # -----------------------------------------------
            # FGlobal propagation
            # -----------------------------------------------

            else:

                input_f_global = (
                    torch.cat(
                        (
                            v_at.repeat(
                                len(
                                    high_cosine_indices
                                ),
                                1,
                            ),

                            self.current_state[
                                0,
                                action,
                                :
                            ].repeat(
                                len(
                                    high_cosine_indices
                                ),
                                1,
                            ),

                            self.current_state[
                                0,
                                high_cosine_indices,
                                :
                            ],
                        ),
                        dim=1,
                    )
                )

                self.current_state[
                    0,
                    high_cosine_indices,
                    :
                ] = state_update_net(
                    input_f_global
                )

                self.current_state[
                    0,
                    action,
                    :
                ] = v_at

            # -----------------------------------------------
            # Explicitly selected patches ALWAYS contain
            # their true HR representations.
            # -----------------------------------------------

            if (
                len(
                    self.visited_patch_idx
                )
                > 0
            ):

                idx = torch.tensor(
                    self.visited_patch_idx,
                    device=device,
                    dtype=torch.long,
                )

                self.current_state[
                    0,
                    idx,
                    :
                ] = self.hr_features[
                    idx
                ]

            # -----------------------------------------------
            # SSL reward
            # -----------------------------------------------

            loss_after = (
                self.ssl_state_loss(
                    self.current_state
                )
            )

            reward = (
                loss_before
                - loss_after
            ).item()

            # -----------------------------------------------
            # Terminal condition
            # -----------------------------------------------

            done = (
                self.current_time_step
                >= self.max_time_steps
            )

            if done:

                final_state = (
                    self.current_state.clone()
                )

                self.reset()

                return (
                    final_state,
                    reward,
                    done,
                )

            # -----------------------------------------------
            # All regions have now been covered
            # -----------------------------------------------

            if (
                self.visited_patches
                .sum()
                .item()
                == self.N
            ):

                self.visited_all_patches = (
                    True
                )

            return (
                self.current_state.clone(),
                reward,
                done,
            )

        # Defensive fallback.
        return (
            self.current_state.clone(),
            0.0,
            True,
        )


    def get_similar_patches(self):
        return self.high_cosine_indices