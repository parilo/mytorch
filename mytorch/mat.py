import numpy as np

from mytorch.common import Op, Tensor, BackwardOp


class MatVecMul(Op):

    def fwd(self, mat: Tensor, vectors: Tensor):
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
