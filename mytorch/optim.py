from typing import List

import numpy as np

from mytorch.core import Tensor


class Adam:

    def __init__(
            self,
            params: List[Tensor],
            lr: float,
            beta1: float = 0.9,
            beta2: float = 0.999,
            epsilon: float = 1e-8,
    ):
        self.params = params
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.v_params = [np.zeros_like(pm.data) for pm in self.params]
        self.s_params = [np.zeros_like(pm.data) for pm in self.params]
        self.t = 1.0

    def zero_grad(self):
        for pm in self.params:
            pm.zero_grad()

    def step(self):
        for i in range(len(self.params)):
            pm = self.params[i]
            if not pm.grad:
                raise RuntimeError('Params should have gradients')
            self.v_params[i] = self.beta1 * self.v_params[i] + (1 - self.beta1) * pm.grad.data
            vpm = self.v_params[i] / (1 - self.beta1 ** self.t)
            self.s_params[i] = self.beta2 * self.s_params[i] + (1 - self.beta2) * pm.grad.data * pm.grad.data
            spm = self.s_params[i] / (1 - self.beta2 ** self.t)
            pm.data -= self.lr * vpm / (np.sqrt(spm) + self.epsilon)
            self.t += 1
