import numpy as np


class BackwardOp:

    def __init__(self, op, *args, **kwargs):
        self.op = op
        self.args = args
        self.kwargs = kwargs

    def __call__(self, grad=None):
        if grad is None:
            grad = Tensor(np.array([1.0])[0])
        self.op.bwd(grad, *self.args, **self.kwargs)

    def __repr__(self):
        return f'Backward: {self.op}' + \
               ''.join([arg.data.shape for arg in self.args] if self.args else []) + \
               ''.join([f' {name} {val.data.shape}' for name, val in self.kwargs.items()] if self.kwargs else [])


class Tensor:

    def __init__(
            self,
            data: np.ndarray,
            requires_grad: bool = False,
            backward=None,
    ):
        self.requires_grad = requires_grad
        self.data = np.copy(data)
        self.backward = backward or BackwardOp(self)
        self.grad = None

    def bwd(self, grad):
        grad_data = grad.data
        # sum grad over batch if needed
        if len(grad_data.shape) > len(self.data.shape):
            grad_data = grad_data.sum(0)
        if not self.grad:
            self.grad = Tensor(np.zeros_like(self.data))
        self.grad.data += grad_data

    def __repr__(self):
        return str(self.data)


class Op:

    def __call__(self, *args, **kwargs) -> Tensor:
        return self.fwd(*args, **kwargs)

    def fwd(self, *args, **kwargs) -> Tensor:
        raise NotImplemented()

    def bwd(self, *args, **kwargs):
        raise NotImplemented()
