import numpy as np

from mytorch.core import Op, Tensor, BackwardOp


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
        num_items = np.prod(input.data.shape)
        if input.requires_grad:
            grad = 2 * input_grad.data * (input.data - target.data) / num_items
            input.backward(Tensor(grad))

        if target.requires_grad:
            grad = 2 * input_grad.data * (target.data - input.data) / num_items
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


class Slice(Op):

    def fwd(self, tensor: Tensor, slices):
        return Tensor(
            data=tensor.data[slices],
            requires_grad=tensor.requires_grad,
            backward=BackwardOp(self, tensor=tensor, slices=slices),
        )

    def bwd(self, input_grad: Tensor, tensor: Tensor, slices):
        if tensor.requires_grad:
            grad = np.zeros_like(tensor.data)
            grad[slices] = input_grad.data
            tensor.backward(Tensor(grad))
