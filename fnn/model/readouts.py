import torch
from torch.nn import functional
from .elements import Dropout
from .parameters import Parameter, ParameterList
from .modules import Module, ModuleList
from .utils import to_groups_2d


# -------------- Readout Prototype --------------


class Readout(Module):
    """Readout Module"""

    def _init(self, cores, readouts, units, streams):
        """
        Parameters
        ----------
        cores : int
            core channels per stream (C)
        readouts : int
            readouts per unit and stream (R)
        units : int
            number of units (U)
        streams : int
            number of streams (S)
        """
        raise NotImplementedError()

    def forward(self, core, stream=None):
        """
        Parameters
        ----------
        core : Tensor
            [N, S*C, H, W] -- stream is None
                or
            [N, C, H, W] -- stream is int
        stream : int | None
            specific stream | all streams

        Returns
        -------
        Tensor
            [N, S, U, R] -- stream is None
                or
            [N, U, R] -- stream is int
        """
        raise NotImplementedError()


# -------------- Readout Types --------------


class PositionFeature(Readout):
    """Position & Feature Readout"""

    def __init__(self, position, bound, feature):
        """
        Parameters
        ----------
        position : fnn.model.positions.Position
            spatial position
        bounds : fnn.model.bounds.Bound
            spatial bound
        feature : fnn.model.features.Feature
            feature weights
        """
        super().__init__()
        self.position = position
        self.bound = bound
        self.feature = feature

    def _init(self, cores, readouts, units, streams):
        """
        Parameters
        ----------
        cores : int
            core channels per stream (C)
        readouts : int
            readouts per unit and stream (O)
        units : int
            number of units (U)
        streams : int
            number of streams (S)
        """
        self.cores = int(cores)
        self.readouts = int(readouts)
        self.units = int(units)
        self.streams = int(streams)

        self.drop = ModuleList([Dropout() for _ in range(self.streams + 1)])

        self.position._init(
            units=self.units,
        )
        self.feature._init(
            inputs=self.cores,
            outputs=self.readouts,
            units=self.units,
            streams=self.streams,
        )
        bias = lambda: Parameter(torch.zeros(self.units, self.readouts))
        self.biases = ParameterList([bias() for _ in range(self.streams)])
        self.biases.scale = self.units
        self.biases.decay = False
        self.biases.norm_dim = 0

    def forward(self, core, stream=None):
        """
        Parameters
        ----------
        core : Tensor
            [N, S*C, H, W] -- stream is None
                or
            [N, C, H, W] -- stream is int
        stream : int | None
            specific stream | all streams

        Returns
        -------
        Tensor
            [N, S, U, R] -- stream is None
                or
            [N, U, R] -- stream is int
        """
        if self.training:
            position = self.position.sample(core.size(0))
        else:
            position = self.position.mean.expand(core.size(0), -1, -1)

        if stream is None:
            drop = self.drop[self.streams]
        else:
            drop = self.drop[stream]

        out = functional.grid_sample(
            drop(core),
            grid=self.bound(position).unsqueeze(dim=2),
            mode="bilinear",
            padding_mode="border",
            align_corners=False,
        )
        feature = self.feature(stream=stream)

        if stream is None:
            out = to_groups_2d(out, self.streams).squeeze(dim=4)
            out = torch.einsum("S U R C , N S C U -> N S U R", feature, out)
            out = out + torch.stack(list(self.biases), dim=0)

        else:
            out = out.squeeze(dim=3)
            out = torch.einsum("U R C , N C U -> N U R", feature, out)
            out = out + self.biases[stream]

        return out
