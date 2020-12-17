from .tensor import Tensor


class MSELoss:
    def __call__(self, targets: Tensor, predictions: Tensor):
        return (targets - predictions) ** 2 / targets.shape[0]