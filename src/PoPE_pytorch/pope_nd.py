# ruff: noqa: DOC501
# pyright: reportUntypedFunctionDecorator=none
"""Access multi-axis extensions of Polar Coordinate Positional Embedding.

(AI generated docstring)

You can use this module to extend the one-dimensional PoPE construction from the paper to
grid-structured positions such as time-frequency lattices. The module splits the rotated feature
dimension across axes, constructs a separate frequency bank for each axis, and concatenates those
phases into a single PoPE carrier compatible with the attention helpers.

Contents
--------
Classes
	AxialPoPE
		Generalize PoPE from one positional axis to multiple positional axes.

References
----------
[1] Gopalakrishnan, A., Csordás, R., Schmidhuber, J., and Mozer, M. C.
	(2026). Decoupling the ``What'' and ``Where'' With Polar Coordinate Positional Embedding. Local
	project manuscript at `Z0Z_notes/Polar_Coordinate_Positional_Embedding/iclr2026_conference.tex`.
"""

from __future__ import annotations

from einops import einsum
from hunterMakesPy import raiseIfNone
from math import pi
from PoPE_pytorch import apply_pope_to_qk, PolarEmbedReturn
from torch import arange, cat, meshgrid, stack, Tensor
from torch.amp.autocast_mode import autocast
from torch.nn import Module, Parameter, ParameterList
from torch_einops_kit import exists
from typing import TYPE_CHECKING
import torch

if TYPE_CHECKING:
	from torch._prims_common import DeviceLikeType
	from torch.types import Number

class AxialPoPE(Module):
	"""Instantiate multi-axis PoPE embeddings for grid-structured positions.

	(AI generated docstring)

	You can use this module to generalize the paper's one-dimensional PoPE construction [1] to
	positions with several axes. `AxialPoPE` assigns a separate geometric frequency bank to each axis
	in `axial_dims`, concatenates the resulting phases, and learns the same per-head key-side bias
	used by the one-dimensional implementation.

	Parameters
	----------
	dim : int
		Total rotated feature dimension after concatenating all axes.
	heads : int
		Number of attention heads with independent key-side biases.
	axial_dims : tuple[int, ...] | None = None
		Split of `dim` across positional axes. When `None`, the module behaves like a single-axis
		PoPE.
	theta : float = 10000
		Base wavelength used for each axis-specific frequency bank.
	bias_uniform_init : bool = False
		Initialize the learnable bias from `Uniform(-2π, 0)` instead of zero.

	Attributes
	----------
	dim : int
		Total rotated feature dimension.
	heads : int
		Number of attention heads.
	axial_dims : tuple[int, ...]
		Feature allocation for each positional axis.
	inv_freqs : ParameterList
		Axis-specific frequency banks.
	bias : Parameter
		Per-head phase offsets shared across all axes after concatenation.

	See Also
	--------
	separate.pope.PoPE : One-dimensional PoPE module. separate.pope.apply_pope_to_qk : Consume the
	output of `forward`.

	Paper Mapping
	-------------
	The paper defines PoPE for a single sequence axis [1]. `AxialPoPE` preserves the same
	decomposition into magnitudes and position-only phases, but replaces the scalar position with a
	coordinate tuple and concatenates one PoPE phase bank per axis.

	References
	----------
	[1] Gopalakrishnan, A., Csordás, R., Schmidhuber, J., and Mozer, M. C.
		(2026). Decoupling the ``What'' and ``Where'' With Polar Coordinate Positional Embedding.
		Local project manuscript at
		`Z0Z_notes/Polar_Coordinate_Positional_Embedding/iclr2026_conference.tex`.
	"""

	apply_pope_to_qk = staticmethod(apply_pope_to_qk)

	def __init__(
		self, dim: int, *, heads: int, axial_dims: tuple[int, ...] | None = None, theta: float = 10000, bias_uniform_init: bool = False
	) -> None:
		super().__init__()
		self.dim: int = dim
		self.heads: int = heads

		if not exists(axial_dims):
			axial_dims = (dim,)

		self.axial_dims: tuple[int, ...] = raiseIfNone(axial_dims)
		if sum(self.axial_dims) != dim:
			message: str = f'I received `{self.axial_dims = }` and `{dim = }`, but I need `sum(self.axial_dims) == dim`.'
			raise ValueError(message)

		# inv freqs for each axial dimension

		self.inv_freqs = ParameterList()

		for axial_dim in self.axial_dims:
			inv_freqs = theta ** -(arange(axial_dim).float() / axial_dim)
			self.inv_freqs.append(Parameter(inv_freqs, requires_grad=False))

		# the learned bias on the keys

		self.bias = Parameter(torch.zeros(heads, dim))

		if bias_uniform_init:
			self.bias.uniform_(-2.0 * pi, 0.0)

	@property
	def device(self) -> torch.device:
		"""Retrieve the device hosting the learned axial bias tensor.

		(AI generated docstring)

		This property exposes the device used by `bias` so callers can construct grid coordinates on
		the same device before calling `forward`.

		Returns
		-------
		device : torch.device
			Device holding `bias`.
		"""
		return self.bias.device

	@staticmethod
	def get_grid_positions(*dims: Number, device: DeviceLikeType | None = None) -> Tensor:
		"""Generate flattened coordinate tuples for an axial grid.

		(AI generated docstring)

		This method constructs a dense grid of integer coordinates and flattens the grid into a list
		of positions suitable for `forward`. The last axis varies fastest because the method reverses
		the `meshgrid` tuple before stacking.

		Parameters
		----------
		*dims : Number
			Extent of each positional axis.
		device : DeviceLikeType | None = None
			Device on which to allocate the coordinate tensor.

		Returns
		-------
		positions : Tensor
			Tensor of flattened coordinate tuples with shape `(∏ dims, len(dims))`.
		"""
		grid: tuple[Tensor, ...] = meshgrid(*[arange(d, device=device).float() for d in dims], indexing='ij')
		return stack([g.flatten() for g in reversed(grid)], dim=-1)

	@autocast('cuda', enabled=False)
	def forward(self, pos_or_dims: Tensor | tuple[int, ...]) -> tuple[Tensor, Tensor]:
		"""Generate axial PoPE phases and biases for explicit or implicit positions.

		(AI generated docstring)

		This method accepts either an explicit coordinate tensor or an axis shape, constructs one
		phase bank per axis, concatenates those phase banks, and returns the resulting PoPE carrier
		together with the clamped key-side bias.

		Parameters
		----------
		pos_or_dims : Tensor | tuple[int, ...]
			Explicit coordinates or the axis extents used to synthesize a full grid.

		Returns
		-------
		polarEmbed : tuple[Tensor, Tensor]
			Tuple containing concatenated axial phases and the clamped per-head bias.

		See Also
		--------
		get_grid_positions : Generate the implicit coordinates accepted by this method.
		separate.pope.apply_pope_to_qk : Consume the returned tuple inside attention code.

		Mathematical Basis
		------------------
		For each axis `a`, the method forms `freqs_a = pos[..., a] θ_a` using that axis's frequency
		bank, then concatenates the axis-specific phase tensors in feature space. The returned `bias`
		plays the same role as `δ_c` in the biased one-dimensional PoPE formulation [1].

		References
		----------
		[1] Gopalakrishnan, A., Csordás, R., Schmidhuber, J., and Mozer, M. C.
			(2026). Decoupling the ``What'' and ``Where'' With Polar Coordinate Positional Embedding.
			Local project manuscript at
			`Z0Z_notes/Polar_Coordinate_Positional_Embedding/iclr2026_conference.tex`.
		"""
		# handle auto grid generation if tuple is passed

		if isinstance(pos_or_dims, tuple):
			pos: Tensor = self.get_grid_positions(*pos_or_dims, device=self.device)
		else:
			pos = pos_or_dims

		# pos shape is (..., N) where N is len(axial_dims)

		if pos.shape[-1] != len(self.axial_dims):
			message: str = (
				f'I received `{pos.shape[-1] = }` and `{len(self.axial_dims) = }`, '
				'but I need `pos.shape[-1] == len(self.axial_dims)`.'
			)
			raise ValueError(message)

		all_freqs: list[Tensor] = []

		for i, inv_freqs in enumerate(self.inv_freqs):
			# pos_i shape is (...)

			pos_i: Tensor = pos[..., i]

			# freqs_i shape is (..., axial_dim)

			freqs_i: Tensor = einsum(pos_i, inv_freqs, '... , d -> ... d')
			all_freqs.append(freqs_i)

		# concat axial freqs

		freqs: Tensor = cat(all_freqs, dim=-1)

		# the bias, with clamping

		bias: Tensor = self.bias.clamp(-2.0 * pi, 0.0)

		return PolarEmbedReturn(freqs, bias)
