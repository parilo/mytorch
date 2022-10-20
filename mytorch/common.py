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
        self.grad = grad
        # sum grad over batch if needed
        if len(self.grad.data.shape) > len(self.data.shape):
            self.grad = Tensor(self.grad.data.sum(0))

    def __repr__(self):
        return str(self.data)


class Op:

    def __call__(self, *args, **kwargs):
        return self.fwd(*args, **kwargs)

    def fwd(self, *args, **kwargs):
        raise NotImplemented()

    def bwd(self, *args, **kwargs):
        raise NotImplemented()


class MSE(Op):

    def fwd(self, input: Tensor, target: Tensor) -> Tensor:
        diff = input.data - target.data
        mse_data = (diff * diff).mean()
        return Tensor(
            data=mse_data,
            requires_grad=input.requires_grad or target.requires_grad,
            backward=BackwardOp(self, input=input, target=target),
        )

    def bwd(self, input_grad: Tensor, input: Tensor, target: Tensor):
        inp_shape = input.data.shape
        if input.requires_grad:
            grad = 2 * input_grad.data * (input.data - target.data) / inp_shape[0] / inp_shape[1]
            input.backward(Tensor(grad))

        if target.requires_grad:
            grad = 2 * input_grad.data * (target.data - input.data) / inp_shape[0] / inp_shape[1]
            target.backward(Tensor(grad))


class Add(Op):

    def fwd(self, tensor1: Tensor, tensor2: Tensor):
        return Tensor(
            data=tensor1.data + tensor2.data,
            requires_grad=tensor1.requires_grad or tensor2.requires_grad,
            backward=BackwardOp(self, tensor1=tensor1, tensor2=tensor2),
        )

    def bwd(self, input_grad: Tensor, tensor1: Tensor, tensor2: Tensor):
        if tensor1.requires_grad:
            grad = input_grad.data * np.ones_like(tensor1.data)
            tensor1.backward(Tensor(grad))

        if tensor2.requires_grad:
            grad = input_grad.data * np.ones_like(tensor2.data)
            tensor2.backward(Tensor(grad))
