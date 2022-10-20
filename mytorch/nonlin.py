import numpy as np

from mytorch.common import Op, Tensor, BackwardOp


class ReLU(Op):

    def fwd(self, tensor: Tensor):
        return Tensor(
            data=np.where(tensor.data > 0, tensor.data, 0),
            requires_grad=tensor.requires_grad,
            backward=BackwardOp(self, tensor=tensor),
        )

    def bwd(self, input_grad: Tensor, tensor: Tensor):
        if tensor.requires_grad:
            grad = input_grad.data * np.where(tensor.data > 0, 1, 0)
            tensor.backward(Tensor(grad))
