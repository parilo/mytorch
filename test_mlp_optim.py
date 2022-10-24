import torch as t
import torch.nn.functional as F
import numpy as np

from mytorch.common import MSE, Add
from mytorch.core import Tensor
from mytorch.mat import MatVecMul
from mytorch.nonlin import ReLU
from mytorch.optim import Adam


def main():

    bias1 = Tensor(np.random.randn(20), requires_grad=True)
    weights1 = Tensor(np.random.randn(20, 2), requires_grad=True)

    bias2 = Tensor(np.random.randn(2), requires_grad=True)
    weights2 = Tensor(np.random.randn(2, 20), requires_grad=True)

    mse = MSE()
    add = Add()
    matvecmul = MatVecMul()
    relu = ReLU()

    optim = Adam([weights1, bias1, weights2, bias2], lr=1e-4)

    for i in range(20000):

        x_gt = Tensor(np.random.randn(64, 2))
        y_gt = Tensor(x_gt.data * x_gt.data)

        y = add(matvecmul(weights1, x_gt), bias1)
        y = relu(y)
        y = add(matvecmul(weights2, y), bias2)

        loss = mse(y, y_gt)
        print(f'--- {i} loss {loss.data}')

        optim.zero_grad()
        loss.backward()
        optim.step()


if __name__ == '__main__':
    main()
