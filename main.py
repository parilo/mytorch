import torch as t
import torch.nn.functional as F
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


class MatVecMul(Op):

    def fwd(self, mat: Tensor, vectors: Tensor):
        print(f'mat.data {mat.data.shape}')
        print(f'vectors.data.transpose() {vectors.data.transpose().shape}')
        return Tensor(
            data=np.matmul(mat.data, vectors.data.transpose()).transpose(),
            requires_grad=mat.requires_grad or vectors.requires_grad,
            backward=BackwardOp(self, mat=mat, vectors=vectors),
        )

    def bwd(self, input_grad: Tensor, mat: Tensor, vectors: Tensor):
        if mat.requires_grad:
            grad = np.expand_dims(input_grad.data, axis=-1) * \
                   np.stack([vectors.data] * input_grad.data.shape[1], axis=1)
            mat.backward(Tensor(grad))

        if vectors.requires_grad:
            grad = (np.expand_dims(input_grad.data, axis=-1) * mat.data).sum(-2)
            vectors.backward(Tensor(grad))


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


def main():

    x_gt = Tensor(np.random.randn(10, 2), requires_grad=True)
    y_gt = Tensor(x_gt.data * x_gt.data)

    bias1 = Tensor(np.random.randn(20), requires_grad=True)
    weights1 = Tensor(np.random.randn(20, 2), requires_grad=True)

    bias2 = Tensor(np.random.randn(2), requires_grad=True)
    weights2 = Tensor(np.random.randn(2, 20), requires_grad=True)

    mse = MSE()
    add = Add()
    matvecmul = MatVecMul()
    relu = ReLU()

    y = add(matvecmul(weights1, x_gt), bias1)
    y = relu(y)
    y = add(matvecmul(weights2, y), bias2)

    loss = mse(y, y_gt)
    loss.backward()

    # pytorch
    tx_gt, ty_gt = t.as_tensor(x_gt.data), t.as_tensor(y_gt.data)
    tbias1 = t.as_tensor(bias1.data)
    tweights1 = t.as_tensor(weights1.data)
    tbias2 = t.as_tensor(bias2.data)
    tweights2 = t.as_tensor(weights2.data)
    tx_gt.requires_grad = True
    tweights1.requires_grad = True
    tbias1.requires_grad = True
    tweights2.requires_grad = True
    tbias2.requires_grad = True

    ty = t.matmul(
        tweights1,
        tx_gt.transpose(1, 0)
    ).transpose(1, 0) + tbias1
    ty = F.relu(ty)
    ty = t.matmul(
        tweights2,
        ty.transpose(1, 0)
    ).transpose(1, 0) + tbias2

    tloss = F.mse_loss(ty, ty_gt)
    tloss.backward()

    print(bias2.grad)
    print(tbias2.grad)

    print(f'loss {np.allclose(loss.data, tloss.detach().numpy())}')
    print(f'weights1 {np.allclose(weights1.grad.data, tweights1.grad.numpy())}')
    print(f'bias1 {np.allclose(bias1.grad.data, tbias1.grad.numpy())}')
    print(f'weights1 {np.allclose(weights1.grad.data, tweights1.grad.numpy())}')
    print(f'weights2 {np.allclose(weights2.grad.data, tweights2.grad.numpy())}')
    print(f'bias2 {np.allclose(bias2.grad.data, tbias2.grad.numpy())}')
    print(f'weights2 {np.allclose(weights2.grad.data, tweights2.grad.numpy())}')


if __name__ == '__main__':
    main()
