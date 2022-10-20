import torch as t
import torch.nn.functional as F
import numpy as np

from mytorch.common import Tensor, MSE, Add
from mytorch.mat import MatVecMul
from mytorch.nonlin import ReLU


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

    print(f'loss {np.allclose(loss.data, tloss.detach().numpy())}')
    print(f'weights1 {np.allclose(weights1.grad.data, tweights1.grad.numpy())}')
    print(f'bias1 {np.allclose(bias1.grad.data, tbias1.grad.numpy())}')
    print(f'weights1 {np.allclose(weights1.grad.data, tweights1.grad.numpy())}')
    print(f'weights2 {np.allclose(weights2.grad.data, tweights2.grad.numpy())}')
    print(f'bias2 {np.allclose(bias2.grad.data, tbias2.grad.numpy())}')
    print(f'weights2 {np.allclose(weights2.grad.data, tweights2.grad.numpy())}')


if __name__ == '__main__':
    main()
