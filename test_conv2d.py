import torch as t
import torch.nn.functional as F
import numpy as np

from mytorch.conv2d import Conv2dStep, Conv2d
from mytorch.core import Tensor
from mytorch.common import MSE, Add, Slice
from mytorch.nonlin import ReLU


def main():

    x_gt = Tensor(np.random.randn(10, 3, 32, 32), requires_grad=True)
    # y_gt = Tensor(x_gt.data * x_gt.data)
    # y_gt = Tensor(np.ones((10, 8, 14, 14)))
    y_gt = Tensor(np.ones((10, 3, 5, 5)))

    # out channels, in channels, filter width, filter height
    filters1 = Tensor(np.random.randn(8, 3, 5, 5), requires_grad=True)
    bias1 = Tensor(np.random.randn(8), requires_grad=True)

    bias2 = Tensor(np.random.randn(3), requires_grad=True)
    filters2 = Tensor(np.random.randn(3, 8, 5, 5), requires_grad=True)

    conv2d_step = Conv2dStep()
    conv2d = Conv2d()
    mse = MSE()
    add = Add()
    relu = ReLU()
    slice_op = Slice()

    y = conv2d(
        input_features=x_gt,
        filters=filters1,
        bias=bias1,
        stride=2,
        conv2d_step_op=conv2d_step,
        bias_add_op=add,
        slice_op=slice_op,
    )
    y = relu(y)
    y = conv2d(
        input_features=y,
        filters=filters2,
        bias=bias2,
        stride=2,
        conv2d_step_op=conv2d_step,
        bias_add_op=add,
        slice_op=slice_op,
    )
    print('y shape', y.data.shape)

    loss = mse(y, y_gt)
    loss.backward()

    # pytorch
    tx_gt, ty_gt = t.as_tensor(x_gt.data), t.as_tensor(y_gt.data)
    tbias1 = t.as_tensor(bias1.data)
    tfilters1 = t.as_tensor(filters1.data)
    tbias2 = t.as_tensor(bias2.data)
    tfilters2 = t.as_tensor(filters2.data)
    tx_gt.requires_grad = True
    tfilters1.requires_grad = True
    tbias1.requires_grad = True
    tfilters2.requires_grad = True
    tbias2.requires_grad = True

    ty = F.conv2d(input=tx_gt, weight=tfilters1, bias=tbias1, stride=2)
    ty = F.relu(ty)
    ty = F.conv2d(input=ty, weight=tfilters2, bias=tbias2, stride=2)

    tloss = F.mse_loss(ty, ty_gt)
    tloss.backward()

    # print('bias1', bias1.grad.data)
    # print('tbias1', tbias1.grad.numpy())

    print(f'y {np.allclose(y.data, ty.detach().numpy())}')
    print(f'loss {np.allclose(loss.data, tloss.detach().numpy())}')
    print(f'x grad {np.allclose(x_gt.grad.data, tx_gt.grad.numpy())}')
    print(f'filters1 {np.allclose(filters1.grad.data, tfilters1.grad.numpy())}')
    print(f'bias1 {np.allclose(bias1.grad.data, tbias1.grad.numpy())}')
    print(f'bias2 {np.allclose(bias2.grad.data, tbias2.grad.numpy())}')
    print(f'filters {np.allclose(filters2.grad.data, tfilters2.grad.numpy())}')


if __name__ == '__main__':
    main()
