from typing import List

import torch as t
import torch.nn.functional as F
import numpy as np

from mytorch.common import Op, Tensor, BackwardOp, MSE, Add


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
            input_features.bwd(Tensor(grad))

        if filters.requires_grad:
            grad = np.expand_dims(input_grad.data, axis=(2, 3, 4)) * \
                   np.expand_dims(input_features.data, axis=1)
            filters.bwd(Tensor(grad))


class ToFeatureMap(Op):

    def fwd(self, tensors: List[List[Tensor]]) -> Tensor:
        data = np.zeros(tuple(tensors[0][0].data.shape) + (len(tensors), len(tensors[0])))
        requires_grad = False
        for rows in tensors:
            for item in rows:
                data[:, :] = item.data
                requires_grad |= item.requires_grad
        return Tensor(
            data=data,
            requires_grad=requires_grad,
            backward=BackwardOp(self, tensors=tensors) if requires_grad else None,
        )

    def bwd(self, input_grad: Tensor, tensors: List[List[Tensor]]):
        for rows in enumerate(tensors):
            for item in rows:
                if item.requires_grad:
                    item.bwd(input_grad[:, :, ])


class Conv2d(Op):

    def fwd(
            self,
            input_features: Tensor,
            filters: Tensor,
            bias: Tensor,
            stride: int,
            conv2d_step_op: Op,
            bias_add_op: Op,
            to_feature_map_op: Op,
    ) -> Tensor:
        features_shape = input_features.data.shape
        fh, fw = filters.data.shape[2:4]
        convs = []
        for i in range(0, features_shape[2] - fh, stride):
            convs.append([])
            for j in range(0, features_shape[3] - fw, stride):
                convs[-1].append(
                    bias_add_op(
                        conv2d_step_op(input_features.data[:, :, i:i + fh, j:j + fw], filters),
                        bias
                    )
                )
        return to_feature_map_op(convs)

    def bwd(self, input_grad: Tensor, input_features: Tensor, filters: Tensor):
        pass


def main():

    x_gt = Tensor(np.random.randn(10, 3, 5, 5), requires_grad=True)
    # y_gt = Tensor(x_gt.data * x_gt.data)
    y_gt = Tensor(np.ones((10, 8)))

    # out channels, in channels, filter width, filter height
    filters1 = Tensor(np.random.randn(8, 3, 5, 5), requires_grad=True)
    bias1 = Tensor(np.random.randn(8), requires_grad=True)

    # bias2 = Tensor(np.random.randn(2), requires_grad=True)
    # weights2 = Tensor(np.random.randn(2, 20), requires_grad=True)

    conv2d_step = Conv2dStep()
    mse = MSE()
    add = Add()
    # relu = ReLU()

    y = conv2d_step(Tensor(x_gt.data[:, :, :5, :5], requires_grad=True), filters1)
    y = add(y, bias1)

    loss = mse(y, y_gt)
    loss.backward()

    # print(y)

    # pytorch
    tx_gt, ty_gt = t.as_tensor(x_gt.data), t.as_tensor(y_gt.data)
    tbias1 = t.as_tensor(bias1.data)
    tfilters1 = t.as_tensor(filters1.data)
    # tbias2 = t.as_tensor(bias2.data)
    # tweights2 = t.as_tensor(weights2.data)
    # tx_gt.requires_grad = True
    tfilters1.requires_grad = True
    tbias1.requires_grad = True
    # tweights2.requires_grad = True
    # tbias2.requires_grad = True

    ty = F.conv2d(tx_gt[:, :, :5, :5], tfilters1).squeeze()
    ty += tbias1

    # print(ty)

    # ty = t.matmul(
    #     tweights1,
    #     tx_gt.transpose(1, 0)
    # ).transpose(1, 0) + tbias1
    # ty = F.relu(ty)
    # ty = t.matmul(
    #     tweights2,
    #     ty.transpose(1, 0)
    # ).transpose(1, 0) + tbias2

    tloss = F.mse_loss(ty, ty_gt)
    tloss.backward()

    # print(bias1.grad)
    # print(tbias1.grad)

    print(f'loss {np.allclose(loss.data, tloss.detach().numpy())}')
    print(f'x {np.allclose(x_gt.data, tx_gt.detach().numpy())}')
    print(f'filters1 {np.allclose(filters1.grad.data, tfilters1.grad.numpy())}')
    print(f'bias1 {np.allclose(bias1.grad.data, tbias1.grad.numpy())}')
    # print(f'weights1 {np.allclose(weights1.grad.data, tweights1.grad.numpy())}')
    # print(f'weights2 {np.allclose(weights2.grad.data, tweights2.grad.numpy())}')
    # print(f'bias2 {np.allclose(bias2.grad.data, tbias2.grad.numpy())}')
    # print(f'weights2 {np.allclose(weights2.grad.data, tweights2.grad.numpy())}')


if __name__ == '__main__':
    main()
