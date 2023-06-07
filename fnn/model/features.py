import torch
from .parameters import Parameter, ParameterList
from .modules import Module


# -------------- Feature Prototype --------------


class Feature(Module):
    """Feature Module"""

    def _init(self, inputs, outputs, units, streams):
        """
        Parameters
        ----------
        inputs : int
            inputs per stream (I)
        outputs : int
            outputs per unit and stream (O)
        units : int
            number of units (U)
        streams : int
            number of streams (S)
        """
        raise NotImplementedError()

    def forward(self, stream=None):
        """
        Parameters
        ----------
        stream : int | None
            specific stream | all streams

        Returns
        -------
        Tensor
            [S, U, O, I] -- stream is None
                or
            [U, O, I] -- stream is int
        """
        raise NotImplementedError()


# -------------- Feature Types --------------


class Norm(Feature):
    """Norm Feature"""

    def __init__(self, eps=1e-5):
        """
        Parameters
        ----------
        eps : float
            small value added to denominator for numerical stability
        """
        super().__init__()
        self.eps = float(eps)
        self._features = dict()

    def _init(self, inputs, outputs, units, streams):
        """
        Parameters
        ----------
        inputs : int
            inputs per stream (I)
        outputs : int
            outputs per unit and stream (O)
        units : int
            number of units (U)
        streams : int
            number of streams (S)
        """
        self.inputs = int(inputs)
        self.outputs = int(outputs)
        self.units = int(units)
        self.streams = int(streams)

        weight = lambda: Parameter(torch.zeros(self.units, self.outputs, self.inputs))
        self.weights = ParameterList([weight() for _ in range(self.streams)])
        self.weights.scale = self.units
        self.weights.norm_dim = 2

        gain = lambda: Parameter(torch.ones(self.units, self.outputs))
        self.gains = ParameterList([gain() for _ in range(self.streams)])
        self.gains.scale = self.units
        self.gains.decay = False
        self.gains.norm_dim = 0

    def _reset(self):
        self._features.clear()

    def features(self, stream):
        """
        Parameters
        ----------
        stream : int
            stream index

        Returns
        -------
        Tensor
            [U, O, I]
        """
        features = self._features.get(stream)

        if features is None:
            weight = self.weights[stream]
            gain = self.gains[stream]

            var, mean = torch.var_mean(weight, dim=2, unbiased=False, keepdim=True)
            scale = (var * self.inputs + self.eps).pow(-0.5)

            features = torch.einsum("U O I , U O -> U O I", (weight - mean) * scale, gain)
            self._features[stream] = features

        return features

    def forward(self, stream=None):
        """
        Parameters
        ----------
        stream : int | None
            specific stream | all streams

        Returns
        -------
        Tensor
            [S, U, O, I] -- stream is None
                or
            [U, O, I] -- stream is int
        """
        if stream is None:
            features = [self.features(stream=s) for s in range(self.streams)]
            features = torch.stack(features, dim=0)

        else:
            features = self.features(stream=stream)

        return features
