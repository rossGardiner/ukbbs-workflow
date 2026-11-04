"""Compact GraFITi-style, missing-aware graph forecaster for UKBBS.

This is an independent, research-oriented adaptation of the sparsity-structure
(time <-> channel) graph in Yalavarthi et al., AAAI 2024, NOT a line-for-line
port or a reproduction of the authors' published numerical results.

Each channel is a BBS square, each time node is a calendar year, and an edge
exists for every *observed* historical count or *requested* forecast. The
network updates time nodes, channel nodes and edge states with masked attention.
No future bird counts, test observation masks or post-origin climate are read.

A single network is trained on blocks of squares for one species. Blocking
bounds memory but does NOT constitute geographical adjacency. The model issues
all requested horizons directly from the frozen training origin (no updates).

PyTorch is optional for the project, but required to import this module.
Install with `python -m pip install torch`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Self
import math

import numpy as np
import pandas as pd

try:
    import torch
    from torch import nn
except ImportError as exc:
    raise ImportError("GraFITi adaptation requires PyTorch: python -m pip install torch") from exc

from .forecasting import ForecastTask, PREDICTION_COLUMN, validate_frame


@dataclass(frozen=True)
class GraFITiConfig:
    context_years: int = 8
    max_horizon: int = 6
    sites_per_graph: int = 8
    latent_dim: int = 48
    attention_heads: int = 4
    layers: int = 2
    epochs: int = 35
    batch_size: int = 16
    learning_rate: float = 0.001
    weight_decay: float = 0.0001
    seed: int = 42
    device: str = "auto"

    def __post_init__(self) -> None:
        for name in ("context_years", "max_horizon", "sites_per_graph", "latent_dim",
                     "attention_heads", "layers", "epochs", "batch_size"):
            n = getattr(self, name)
            if type(n) is not int or n < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.latent_dim % self.attention_heads:
            raise ValueError("latent_dim must be divisible by attention_heads")
        for name in ("learning_rate", "weight_decay"):
            n = getattr(self, name)
            if not np.isfinite(n) or n < 0 or (name == "learning_rate" and n == 0):
                raise ValueError(f"{name} must be finite and {'positive' if name == 'learning_rate' else 'nonnegative'}")
        if type(self.seed) is not int:
            raise ValueError("seed must be an integer")
        if self.device not in ("auto", "cpu", "cuda"):
            raise ValueError("device must be 'auto', 'cpu', or 'cuda'")


class _IncidentAttention(nn.Module):
    """Multi-head attention from one node to its incident graph edges.

    `nodes`: [B,N,H], `edges`: [B,N,K,H], `mask`: [B,N,K].
    Handles graph nodes without any incident edge without NaNs.
    """

    def __init__(self, hidden: int, heads: int):
        super().__init__()
        self.heads = heads
        self.head_dim = hidden // heads
        self.query = nn.Linear(hidden, hidden)
        self.key = nn.Linear(hidden, hidden)
        self.value = nn.Linear(hidden, hidden)
        self.out = nn.Linear(hidden, hidden)
        self.ff = nn.Sequential(nn.Linear(hidden, 2 * hidden), nn.ReLU(), nn.Linear(2 * hidden, hidden))
        self.norm1 = nn.LayerNorm(hidden)
        self.norm2 = nn.LayerNorm(hidden)

    def forward(self, nodes: torch.Tensor, edges: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        b, n, k, h = edges.shape
        q = self.query(nodes).reshape(b, n, self.heads, self.head_dim)
        keys = self.key(edges).reshape(b, n, k, self.heads, self.head_dim)
        values = self.value(edges).reshape(b, n, k, self.heads, self.head_dim)
        logits = torch.einsum("bnhd,bnkhd->bnhk", q, keys) / math.sqrt(self.head_dim)
        included = mask.unsqueeze(2)
        probs = torch.softmax(logits.masked_fill(~included, -1e4), dim=-1)
        # A fully unobserved node has *no* message, rather than uniform weights.
        probs = probs * included
        probs = probs / probs.sum(-1, keepdim=True).clamp(min=1e-9)
        messages = torch.einsum("bnhk,bnkhd->bnhd", probs, values).reshape(b, n, h)
        z = self.norm1(nodes + self.out(messages))
        return self.norm2(z + self.ff(z))


class _SparsityGraphLayer(nn.Module):
    def __init__(self, hidden: int, heads: int):
        super().__init__()
        self.to_time = _IncidentAttention(hidden, heads)
        self.to_channel = _IncidentAttention(hidden, heads)
        self.to_edge = nn.Sequential(nn.Linear(3 * hidden, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
        self.norm = nn.LayerNorm(hidden)

    def forward(self, time: torch.Tensor, channel: torch.Tensor, edge: torch.Tensor,
                mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Gather from the previous layer, then update nodes in parallel.
        updated_time = self.to_time(time, edge, mask)
        updated_channel = self.to_channel(channel, edge.transpose(1, 2), mask.transpose(1, 2))
        t = updated_time.unsqueeze(2).expand_as(edge)
        c = updated_channel.unsqueeze(1).expand_as(edge)
        updated_edge = self.norm(edge + self.to_edge(torch.cat([edge, t, c], dim=-1)))
        updated_edge = updated_edge * mask.unsqueeze(-1)
        return updated_time, updated_channel, updated_edge


class _SparsityGraphNetwork(nn.Module):
    """Time–channel bipartite graph with observation and query edges."""

    def __init__(self, config: GraFITiConfig):
        super().__init__()
        d = config.latent_dim
        self.channel_embedding = nn.Embedding(config.sites_per_graph, d)
        self.time_linear = nn.Linear(1, d)
        self.edge_linear = nn.Linear(2, d)  # normalized log count + query indicator
        self.layers = nn.ModuleList(
            _SparsityGraphLayer(d, config.attention_heads) for _ in range(config.layers)
        )
        self.readout = nn.Linear(3 * d, 1)

    def forward(self, years: torch.Tensor, values: torch.Tensor, observed: torch.Tensor,
                query: torch.Tensor) -> torch.Tensor:
        if values.shape != observed.shape or observed.shape != query.shape:
            raise ValueError("Graph values and edge masks must share [B,T,C] shape")
        if (observed & query).any():
            raise ValueError("An edge cannot be observed and queried simultaneously")
        b, t, c = values.shape
        edges_present = observed | query
        # Time encodings are relative to each sample's forecast origin.
        time = torch.sin(self.time_linear(years.unsqueeze(-1)))
        positions = torch.arange(c, device=values.device)
        channel = self.channel_embedding(positions).unsqueeze(0).expand(b, -1, -1)
        raw = torch.stack([torch.where(observed, values, torch.zeros_like(values)),
                           query.to(values.dtype)], dim=-1)
        edge = torch.relu(self.edge_linear(raw)) * edges_present.unsqueeze(-1)
        for layer in self.layers:
            time, channel, edge = layer(time, channel, edge, edges_present)
        prediction = self.readout(torch.cat([
            edge,
            time.unsqueeze(2).expand_as(edge),
            channel.unsqueeze(1).expand_as(edge),
        ], dim=-1)).squeeze(-1)
        return prediction


def _validate_calendar_years(frame: pd.DataFrame, column: str) -> np.ndarray:
    raw = frame[column].to_numpy(dtype=float)
    if not np.isfinite(raw).all() or (raw != np.floor(raw)).any():
        raise ValueError("GraFITi expects integer calendar years")
    return raw.astype(int)


class GraFITiForecaster:
    """Single-species, fixed-origin graph forecaster compatible with run_forecast.

    Missing historical targets are absent edges; real zeros remain observed
    edges. Future query edges are determined by site and forecast year *only*,
    never by whether their bird target was observed in the test set.

    Each graph has a bounded set of channels (squares), sorted deterministically.
    Channel positions have learned embeddings shared between all site blocks.
    This is a sparsity-graph model, not a geographic-neighbourhood GNN.
    """

    def __init__(self, config: GraFITiConfig | None = None):
        self.config = config or GraFITiConfig()
        self._task: ForecastTask | None = None
        self._network: _SparsityGraphNetwork | None = None
        self._species: str | None = None
        self._origin: int | None = None
        self._groups: list[list[str]] | None = None
        self._counts: dict[tuple[str, int], float] | None = None
        self._mean: float | None = None
        self._scale: float | None = None
        self._device: torch.device | None = None
        self.history_: list[float] = []
        self.n_training_graphs_: int = 0

    def _make_graph(self, group: list[str], origin: int, *, labels: bool):
        """Make one context+future graph using only history at/before origin."""
        assert self._counts is not None and self._mean is not None and self._scale is not None
        cfg = self.config
        ts = cfg.context_years + cfg.max_horizon
        n = cfg.sites_per_graph
        times = np.arange(-cfg.context_years + 1, cfg.max_horizon + 1, dtype="float32")
        values = np.zeros((ts, n), dtype="float32")
        observed = np.zeros((ts, n), dtype=bool)
        query = np.zeros((ts, n), dtype=bool)
        targets = np.zeros((ts, n), dtype="float32")
        scored = np.zeros((ts, n), dtype=bool)
        for i, site in enumerate(group):
            for j, offset in enumerate(times.astype(int)):
                year = origin + offset
                if offset <= 0:
                    value = self._counts.get((site, year))
                    if value is not None:
                        values[j, i] = (np.log1p(value) - self._mean) / self._scale
                        observed[j, i] = True
                else:
                    query[j, i] = True
                    # Training labels are *never* part of graph inputs.
                    if labels:
                        target = self._counts.get((site, year))
                        if target is not None and year <= self._origin:
                            targets[j, i] = (np.log1p(target) - self._mean) / self._scale
                            scored[j, i] = True
        return times, values, observed, query, targets, scored

    def fit(self, train: pd.DataFrame, task: ForecastTask) -> Self:
        validate_frame(train, task, labelled=True)
        if tuple(task.group_cols) != ("OS_GRID_REF", "species"):
            raise ValueError('GraFITi requires group_cols=("OS_GRID_REF", "species")')
        data = train.copy()
        data[task.time_col] = _validate_calendar_years(data, task.time_col)
        species = data["species"].unique()
        if len(species) != 1:
            raise ValueError("Fit one species at a time")
        actual = pd.to_numeric(data[task.target], errors="raise").to_numpy(dtype=float)
        if not np.isfinite(actual[~np.isnan(actual)]).all() or (actual[~np.isnan(actual)] < 0).any():
            raise ValueError("Bird counts must be nonnegative and finite where observed")
        if np.isnan(actual).all():
            raise ValueError("No observed training targets")
        sites = sorted(map(str, data["OS_GRID_REF"].unique()))
        groups = [sites[i:i + self.config.sites_per_graph]
                  for i in range(0, len(sites), self.config.sites_per_graph)]
        observed_data = data.loc[data[task.target].notna(), ["OS_GRID_REF", task.time_col, task.target]]
        counts = {(str(site), int(year)): float(value)
                  for site, year, value in observed_data.itertuples(index=False, name=None)}
        log_count = np.log1p(actual[~np.isnan(actual)])
        self._mean = float(log_count.mean())
        self._scale = max(float(log_count.std()), 1e-4)
        self._origin = int(data[task.time_col].max())
        self._counts = counts
        self._groups = groups
        min_year = int(data[task.time_col].min())
        if self._origin - min_year < self.config.context_years:
            raise ValueError("Not enough training years for configured context_years")

        # Training examples: historical forecast origins; labels are bounded
        # strictly by the last training year, even for multi-step queries.
        samples = []
        for old_origin in range(min_year + self.config.context_years - 1, self._origin):
            for group in groups:
                sample = self._make_graph(group, old_origin, labels=True)
                if sample[-1].any():
                    samples.append(sample)
        if not samples:
            raise ValueError("No historical training graphs with observed future targets")
        years, values, observed, query, targets, scored = (
            torch.from_numpy(np.stack([sample[i] for sample in samples])) for i in range(6)
        )
        if self.config.device == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA requested but unavailable")
        device = torch.device("cuda" if (self.config.device == "cuda" or
                          self.config.device == "auto" and torch.cuda.is_available()) else "cpu")
        # Construct model deterministically without permanently altering caller RNG.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(self.config.seed)
            network = _SparsityGraphNetwork(self.config).to(device)
        optimizer = torch.optim.AdamW(network.parameters(), lr=self.config.learning_rate,
                                      weight_decay=self.config.weight_decay)
        generator = torch.Generator().manual_seed(self.config.seed)
        self.history_ = []
        network.train()
        for _ in range(self.config.epochs):
            order = torch.randperm(len(samples), generator=generator)
            total, weight = 0.0, 0
            for indices in order.split(self.config.batch_size):
                yrs, x, xm, q, y, ym = (v[indices].to(device) for v in
                                           (years, values, observed, query, targets, scored))
                optimizer.zero_grad(set_to_none=True)
                estimate = network(yrs, x, xm, q)
                loss = ((estimate - y).square() * ym).sum() / ym.sum()
                if not torch.isfinite(loss):
                    raise RuntimeError("GraFITi training produced nonfinite loss")
                loss.backward()
                optimizer.step()
                count = int(ym.sum())
                total += float(loss.detach()) * count
                weight += count
            self.history_.append(total / weight)
        network.eval()
        self._network = network
        self._device = device
        self._task = task
        self._species = str(species[0])
        self.n_training_graphs_ = len(samples)
        return self

    def predict(self, future: pd.DataFrame) -> pd.DataFrame:
        if self._network is None or self._task is None or self._groups is None:
            raise RuntimeError("Call fit before predict")
        task = self._task
        validate_frame(future, task, labelled=False)
        if task.target in future.columns:
            raise ValueError("Future must not contain target counts")
        data = future.copy()
        data[task.time_col] = _validate_calendar_years(data, task.time_col)
        if not data["species"].astype(str).eq(self._species).all():
            raise ValueError("Only the trained species can be predicted")
        offset = data[task.time_col] - self._origin
        if not offset.between(1, self.config.max_horizon).all():
            raise ValueError(f"Forecast years must be within origin+1 through origin+{self.config.max_horizon}")
        training_sites = {site for group in self._groups for site in group}
        if not set(map(str, data["OS_GRID_REF"])).issubset(training_sites):
            raise ValueError("Test includes squares absent from training")
        predictions = {}
        with torch.no_grad():
            # Query every site/horizon, regardless of test observation status.
            # Therefore adding/removing a test row never changes a prediction.
            for group in self._groups:
                sample = self._make_graph(group, self._origin, labels=False)
                years, values, observed, query = (
                    torch.from_numpy(arr).unsqueeze(0).to(self._device)
                    for arr in sample[:4]
                )
                estimate = self._network(years, values, observed, query)[0].cpu().numpy()
                for i, site in enumerate(group):
                    for horizon in range(1, self.config.max_horizon + 1):
                        normalized = float(estimate[self.config.context_years + horizon - 1, i])
                        log_value = max(0.0, min(20.0, normalized * self._scale + self._mean))
                        predictions[(site, self._origin + horizon)] = float(np.expm1(log_value))
        result = data[task.keys].copy()
        result[PREDICTION_COLUMN] = [
            predictions[(str(site), int(year))]
            for site, year in zip(data["OS_GRID_REF"], data[task.time_col])
        ]
        return result
