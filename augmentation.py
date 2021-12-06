import numpy as np
import random
from itertools import product, permutations


class BasicPolicy(object):
    def __init__(self, mirror_ratio=0, color_change_ratio=0, is_full_set_colors=False):
        self.indices = list(product([0, 1, 2], repeat=3)) if is_full_set_colors else list(permutations(range(3), 3))
        self.indices.insert(0, [0, 1, 2])  # R,G,B
        self.color_change_ratio = color_change_ratio
        self.mirror_ratio = mirror_ratio

    def __call__(self, img, depth):
        policy_idx = random.randint(0, len(self.indices) - 1)
        if random.uniform(0, 1) >= self.color_change_ratio:
            policy_idx = 0

        img = img[..., list(self.indices[policy_idx])]

        if random.uniform(0, 1) <= self.mirror_ratio:
            img = img[..., ::-1, :]
            depth = depth[..., ::-1, :]

        return img, depth

    def __repr__(self):
        return "Basic Policy"