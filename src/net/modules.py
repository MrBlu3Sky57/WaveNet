"""
The file containing all of the relevant PyTorch blocks
"""

import torch
import torch.nn as nn

class DilatedCausalConv1d(nn.Module):
    # Input should be in shape: (N, K, T) where K is the Channel sizeß
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

class Sample(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        return inp[:, :, -1]

class Output(nn.Module):
    def __init__(self, size: tuple[int]):
        super().__init__()
        self.blocks = nn.Sequential()
        for l1, l2 in zip(size, size[1:]):
            self.blocks.append(Conv1x1(l1, l2))
            self.blocks.append(nn.ReLU())
        self.blocks.pop(-1)
        self.blocks.append(Sample())
        self.blocks.append(nn.Softmax())

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        return self.blocks.forward(inp)

class WaveNet(nn.Module):
    """
    A WaveNet with base 2 exponential dilation
    """
    def __init__(self, channels: tuple[int], kernel_size: tuple[int], skip_size: int, out_size: tuple[int]):
        super().__init__()
        self.blocks = []

        # Big Ugly Loop :(
        for i, (c_in, c_hidden, c_out, k) in enumerate(zip(channels, channels[1:], channels[2:], kernel_size)):
            self.blocks.append(WaveNetBlock(
                in_channels=c_in,
                hidden_channels=c_hidden,
                out_channels=c_out,
                skip_channels=skip_size,
                kernel_size=k,
                dilation=2**i
            ))
        self.out = Output(out_size)
        self.skip_size = skip_size

    def forward(self, inp: torch.Tensor):
        skip_aggregate = torch.zeros((inp.shape[0], self.skip_size, inp.shape[2]))
        for block in self.blocks:
            inp, skip = block.forward(inp)
            skip_aggregate += skip
        return self.out.forward(skip_aggregate)