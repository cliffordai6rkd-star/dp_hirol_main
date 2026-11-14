"""Student-owned visual calibration before the frozen final context norm."""

from torch import nn


class StudentContextAdapter(nn.Module):
    """Tokenwise residual mapping; never reads torque, masks or alpha."""

    def __init__(self, dim, *, kind="linear", hidden_dim=128):
        super().__init__()
        if int(dim) < 1 or int(hidden_dim) < 1:
            raise ValueError("Context adapter dimensions must be positive")
        self.dim = int(dim)
        self.kind = str(kind)
        if self.kind == "linear":
            self.mapping = nn.Linear(self.dim, self.dim)
            output = self.mapping
        elif self.kind == "mlp":
            self.mapping = nn.Sequential(
                nn.Linear(self.dim, int(hidden_dim)), nn.GELU(),
                nn.Linear(int(hidden_dim), self.dim),
            )
            output = self.mapping[-1]
        else:
            raise ValueError("context_adapter.kind must be linear or mlp")
        # Start at the unchanged image path; only the student learns the offset.
        nn.init.zeros_(output.weight)
        nn.init.zeros_(output.bias)

    def forward(self, tokens):
        if tokens.ndim != 3 or tokens.shape[-1] != self.dim:
            raise ValueError(f"Context adapter expects [B, N, {self.dim}] tokens")
        return tokens + self.mapping(tokens)
