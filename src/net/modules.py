"""
The file containing all of the relevant PyTorch blocks
"""

import torch
import torch.nn as nn

class DilatedCausalConv1d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int):
        super().__init__()
        self.conv = torch.nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            dilation=dilation
        )
        self.pad_size = (kernel_size - 1) * (dilation + 1)

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        inp = torch.transpose(inp, 1, 2) # Swap channel and time dimensions
        inp = nn.functional.pad(inp, pad=(self.pad_size, 0), mode='constant', value=0)

        return self.conv.forward(inp)

class GatedActivation(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_act = nn.Sigmoid()
        self.filter_act = nn.Tanh()
    def forward(self, gate_unact: torch.Tensor, filtered_unact: torch.Tensor):
        return self.gate_act.forward(gate_unact) * self.filter_act(filtered_unact)

class Conv1x1(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = torch.nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1
        )

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        return self.conv.forward(inp)

class WaveNetBlock(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int, skip_channels: int,
                kernel_size: int, dilation: int):
        super().__init__()
        self.conv = DilatedCausalConv1d(
                in_channels=in_channels,
                out_channels=hidden_channels,
                kernel_size=kernel_size,
                dilation=dilation
                )
        self.gated_act = GatedActivation()
        self.out_proj = Conv1x1(hidden_channels, out_channels)
        self.skip_proj = Conv1x1(hidden_channels, skip_channels)

    def forward(self, inp: torch.Tensor) -> tuple[torch.Tensor]:
        out = self.conv.forward(inp)
        out = self.gated_act(inp)

        skip = self.skip_proj(out)
        out = self.out_proj(out)

        return (out, skip)

class Output(nn.Module):
    pass

class WaveNet(nn.Module):
    pass
