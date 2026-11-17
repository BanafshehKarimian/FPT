import math

import torch
import torch.nn as nn

### For Intermediate Reward

class WSICosineObservationEnv():
    def __init__(self, lr_features, hr_features, conf):
        self.conf = conf
        self.features = lr_features.clone()
        self.current_state = self.features.clone()
        self.hr_features = hr_features
        self.B, self.N, self.d = self.current_state.shape

        self.current_time_step = 0
        self.max_time_steps = math.ceil(self.N * conf.frac_visit)

        self.visited_patches = torch.zeros(
            size=(self.N,),
            dtype=torch.bool,
            device=conf.device,
        )

        self.visited_patch_idx = []
        self.mask = torch.ones((self.N,), device=conf.device)
        self.cosine_threshold = conf.cosine_threshold
        self.high_cosine_indices = None
        self.visited_all_patches = False

    def reset(self):
        self.current_state = self.features.clone()
        self.current_time_step = 0
        self.visited_patches = torch.zeros(
            size=(self.N,),
            dtype=torch.bool,
            device=self.conf.device,
        )
        self.visited_patch_idx = []
        self.mask = torch.ones((self.N,), device=self.conf.device)
        self.high_cosine_indices = None
        self.visited_all_patches = False

    @torch.no_grad()
    def step(self, action, state_update_net, device):
        if isinstance(action, torch.Tensor):
            action = int(action.item())
        else:
            action = int(action)


        if self.current_time_step < self.max_time_steps:
            self.current_time_step += 1

            self.visited_patches[action] = True
            self.visited_patch_idx.append(action)

            valid_indices = (self.mask == 1).nonzero(as_tuple=False)[:, 0]

            cosine_vector = torch.cosine_similarity(
                self.current_state[0],
                self.current_state[0][action].unsqueeze(0),
                dim=-1,
            )

            high_cosine_indices = (
                torch.abs(cosine_vector) >= self.cosine_threshold
            ).nonzero(as_tuple=False)[:, 0]

            #self.mask[high_cosine_indices] = 0
            #self.visited_patches[high_cosine_indices] = True

            high_cosine_indices = high_cosine_indices[high_cosine_indices != action]

            if high_cosine_indices.numel() > 0:
                valid_set = set(valid_indices.tolist())
                high_cosine_indices = torch.tensor(
                    [idx for idx in high_cosine_indices.tolist() if idx in valid_set],
                    device=device,
                    dtype=torch.long,
                )
            else:
                high_cosine_indices = torch.empty(
                    0,
                    device=device,
                    dtype=torch.long,
                )

            self.high_cosine_indices = high_cosine_indices

            v_at = self.hr_features[action].to(device)

            self.current_state = self.current_state.clone()

            if high_cosine_indices.shape[0] == 0:
                self.current_state[0, action, :] = v_at
            else:
                input_f_global = torch.cat(
                    (
                        v_at.repeat(len(high_cosine_indices), 1),
                        self.current_state[0, action, :].repeat(len(high_cosine_indices), 1),
                        self.current_state[0, high_cosine_indices, :],
                    ),
                    dim=1,
                )

                self.current_state[0, high_cosine_indices, :] = state_update_net(input_f_global)
                self.current_state[0, action, :] = v_at

            if len(self.visited_patch_idx) > 0:
                idx = torch.tensor(
                    self.visited_patch_idx,
                    device=device,
                    dtype=torch.long,
                )
                self.current_state[0, idx, :] = self.hr_features[idx]

            if self.visited_patches.sum().item() == self.N:
                self.visited_all_patches = True


        done = self.current_time_step >= self.max_time_steps

        if done:
            final_state = self.current_state.clone()
            self.reset()
            return final_state, 0, done

        return self.current_state.clone(), 0, done

    def get_similar_patches(self):
        return self.high_cosine_indices