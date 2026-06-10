# ruff: noqa: D100
from __future__ import annotations

from typing import NamedTuple, TYPE_CHECKING

if TYPE_CHECKING:
	from torch import Tensor

class PolarEmbedReturn(NamedTuple):
	"""Store PoPE phases and per-head phase biases.

	(AI generated docstring)

	You can use this named tuple as the carrier returned by the PoPE modules.
	`freqs` stores the position-dependent phases for each rotated feature, and
	`bias` stores the learned key-side offsets that implement the biased PoPE
	variant described in the paper [1].

	Attributes
	----------
	freqs : Tensor
		Tensor of position-dependent phases whose last dimension matches the
		rotated feature dimension.
	bias : Tensor
		Tensor of per-head phase biases clamped to the interval `[-2π, 0]` by
		the generating module.

	References
	----------
	[1] Gopalakrishnan, A., Csordás, R., Schmidhuber, J., and Mozer, M. C.
		(2026). Decoupling the ``What'' and ``Where'' With Polar Coordinate
		Positional Embedding. Local project manuscript at
		`Z0Z_notes/Polar_Coordinate_Positional_Embedding/iclr2026_conference.tex`.
	"""

	freqs: Tensor
	bias: Tensor
