from collections import deque
import numpy as np
import random


class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        s, a, r, s2, d = zip(*batch)
        return (np.vstack(s), np.array(a), np.array(r, dtype=np.float32),
                np.vstack(s2), np.array(d, dtype=np.uint8))

    def __len__(self):
        return len(self.buffer)