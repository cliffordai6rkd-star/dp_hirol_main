"""LSTM branch for independent fixed-length tau_other history windows."""

import torch.nn as nn

from model.tau_other_sequence import TauOtherSequenceModelBase, _model_config


class TauOtherLSTMRegressor(TauOtherSequenceModelBase):
    def __init__(self, config):
        super().__init__(config, architecture="lstm")
        model_config = _model_config(config)
        recurrent_dropout = self.dropout if self.num_layers > 1 else 0.0
        self.recurrent = nn.LSTM(
            input_size=self.input_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            dropout=recurrent_dropout,
            batch_first=True,
        )
        self.head = self._build_head(model_config)

    def forward(self, batch):
        sequence = self._prepare_inputs(batch)
        recurrent_output, _ = self.recurrent(sequence)
        return self._finish_output(batch, self.head(recurrent_output[:, -1]))
