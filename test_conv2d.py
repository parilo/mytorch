from typing import List

import torch as t
import torch.nn.functional as F
import numpy as np

from mytorch.core import Op, Tensor, BackwardOp
from mytorch.common import MSE, Add, Slice
from mytorch.nonlin import ReLU


class Conv2dStep(Op):

    def fwd(self, input_features: Tensor, filters: Tensor) -> Tensor:
        return Tensor(
            data=np.tensordot(input_features.data, filters.data, axes=([1, 2, 3], [1, 2, 3])),
            requires_grad=input_features.requires_grad or filters.requires_grad,
            backward=BackwardOp(self, input_features=input_features, filters=filters),
        )

    def bwd(self, input_grad: Tensor, input_features: Tensor, filters: Tensor):
        if input_features.requires_grad:
            grad = np.expand_dims(input_grad.data, axis=(2, 3, 4)) * filters.data
            grad = grad.sum(1)  # sum over filters for each feature map
            input_features.backward(Tensor(grad))

        if filters.requires_grad:
            grad = np.expand_dims(input_grad.data, axis=(2, 3, 4)) * \
                   np.expand_dims(input_features.data, axis=1)
            filters.backward(Tensor(grad))


class Conv2d(Op):

    def fwd(
            self,
            input_features: Tensor,
            filters: Tensor,
            bias: Tensor,
            stride: int,
            conv2d_step_op: Op,
            bias_add_op: Op,
    ) -> Tensor:
        features_shape = input_features.data.shape
        fh, fw = filters.data.shape[2:4]
        rows = list(range(0, features_shape[2] - fh + 1, stride))
        cols = list(range(0, features_shape[3] - fw + 1, stride))
        data = np.zeros((features_shape[0], filters.data.shape[0]) + (len(rows), len(cols)))
        convs = []
        requires_grad = False
        for i in rows:
            convs.append([])
            for j in cols:
                # features_part = Tensor(
                #     input_features.data[:, :, i:i + fh, j:j + fw],
                #     requires_grad=input_features.requires_grad,
                #     backward=BackwardOp(input_features, slices=(
                #         slice(None),
                #         slice(None),
                #         slice(i, i + fh),
                #         slice(j, j + fw),
                #     )),
                # )
                slice_op = Slice()
                features_part = slice_op(input_features, slices=(
                    slice(None),
                    slice(None),
                    slice(i, i + fh),
                    slice(j, j + fw),
                ))
                val = bias_add_op(
                    conv2d_step_op(
                        features_part,
                        filters
                    ),
                    bias
                )
                data[:, :, i // stride, j // stride] = val.data
                convs[-1].append(val)
                requires_grad |= val.requires_grad
        return Tensor(
            data=data,
            requires_grad=requires_grad,
            backward=BackwardOp(self, convs=convs),
        )

    def bwd(self, input_grad: Tensor, convs: List[List[Tensor]]):
        for i, rows in enumerate(convs):
            for j, item in enumerate(rows):
                if item.requires_grad:
                    item.backward(input_grad.data[:, :, i, j])


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

    y = conv2d(
        input_features=x_gt,
        filters=filters1,
        bias=bias1,
        stride=2,
        conv2d_step_op=conv2d_step,
        bias_add_op=add,
    )
    y = relu(y)
    y = conv2d(
        input_features=y,
        filters=filters2,
        bias=bias2,
        stride=2,
        conv2d_step_op=conv2d_step,
        bias_add_op=add,
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
