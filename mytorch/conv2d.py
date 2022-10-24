from typing import List

import numpy as np

from mytorch.core import Op, Tensor, BackwardOp


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
            slice_op: Op,
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
