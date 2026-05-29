"""Tests for the bucket-padding-aware slicing in ``MultiSampleProtenixLoss``.

The loss terms in :mod:`mosaic.losses.structure_prediction` (BinderTargetPAE,
IPTMLoss, …) slice off ``binder_len`` to get "the rest of the chain" and
treat that as real target tokens.  When ``protenij.padding.pad_features``
inflates features to a bucket size, those slices silently include padding
tokens and contaminate every binder-vs-target loss.

The fix routes the per-sample model output through
``_slice_padded_model_output`` inside ``MultiSampleProtenixLoss.__call__``
when bucket padding was applied.  These tests pin the slicing helper and the
``token_mask``-based detection used by ``real_shapes_from_padded_features``.
"""

from __future__ import annotations

import dataclasses

import jax.numpy as jnp
import numpy as np
import pytest

from mosaic.losses.protenix import (
    _slice_padded_model_output,
    real_shapes_from_padded_features,
)
from mosaic.losses.structure_prediction import (
    BinderTargetPAE,
    IPTMLoss,
    StructureModelOutput,
)


def _padded_output(b: int, a_bucket: int) -> StructureModelOutput:
    return StructureModelOutput(
        distogram_logits=jnp.arange(b * b * 4, dtype=jnp.float32).reshape(b, b, 4),
        distogram_bins=jnp.zeros((4,)),
        plddt=jnp.arange(b, dtype=jnp.float32),
        pae=jnp.arange(b * b, dtype=jnp.float32).reshape(b, b),
        pae_logits=jnp.zeros((b, b, 8)),
        pae_bins=jnp.zeros((8,)),
        structure_coordinates=jnp.arange(a_bucket * 3, dtype=jnp.float32).reshape(
            1, a_bucket, 3
        ),
        backbone_coordinates=jnp.zeros((b, 4, 3)),
        full_sequence=jnp.zeros((b, 20)),
        asym_id=jnp.zeros((b,)),
        residue_idx=jnp.arange(b, dtype=jnp.int32),
        atom37_coords=jnp.zeros((b, 37, 3)),
        atom37_mask=jnp.zeros((b, 37)),
    )


def _embed_real_output_in_padding(
    real: StructureModelOutput,
    *,
    b: int,
    a_bucket: int,
    pae_padding_value: float = 1_000.0,
    pae_logits: jnp.ndarray | None = None,
) -> StructureModelOutput:
    n = real.plddt.shape[0]
    a = real.structure_coordinates.shape[1]
    padded = _padded_output(b=b, a_bucket=a_bucket)
    if pae_logits is None:
        pae_logits = (
            jnp.full((b, b, real.pae_logits.shape[-1]), -50.0)
            .at[:n, :n, :]
            .set(real.pae_logits)
        )
    return dataclasses.replace(
        padded,
        distogram_logits=(
            jnp.full((b, b, real.distogram_logits.shape[-1]), -50.0)
            .at[:n, :n, :]
            .set(real.distogram_logits)
        ),
        distogram_bins=real.distogram_bins,
        plddt=jnp.full((b,), -50.0).at[:n].set(real.plddt),
        pae=jnp.full((b, b), pae_padding_value).at[:n, :n].set(real.pae),
        pae_logits=pae_logits,
        pae_bins=real.pae_bins,
        structure_coordinates=(
            jnp.full((1, a_bucket, 3), -50.0)
            .at[:, :a, :]
            .set(real.structure_coordinates)
        ),
        backbone_coordinates=(
            jnp.full((b, 4, 3), -50.0).at[:n].set(real.backbone_coordinates)
        ),
        full_sequence=jnp.full((b, 20), -50.0).at[:n].set(real.full_sequence),
        asym_id=jnp.full((b,), -1.0).at[:n].set(real.asym_id),
        residue_idx=jnp.full((b,), -1, dtype=real.residue_idx.dtype)
        .at[:n]
        .set(real.residue_idx),
        atom37_coords=jnp.full((b, 37, 3), -50.0).at[:n].set(real.atom37_coords),
        atom37_mask=jnp.zeros((b, 37)).at[:n].set(real.atom37_mask),
    )


def _sequence(binder_len: int) -> jnp.ndarray:
    return jnp.zeros((binder_len, 20), dtype=jnp.float32)


def test_slice_padded_model_output_trims_token_and_atom_axes():
    n, a = 6, 40
    b, a_bucket = 16, 128

    sliced = _slice_padded_model_output(
        _padded_output(b=b, a_bucket=a_bucket),
        n_real_tokens=n,
        n_real_atoms=a,
    )

    assert sliced.plddt.shape == (n,)
    assert sliced.pae.shape == (n, n)
    assert sliced.pae_logits.shape == (n, n, 8)
    assert sliced.distogram_logits.shape == (n, n, 4)
    assert sliced.backbone_coordinates.shape == (n, 4, 3)
    assert sliced.full_sequence.shape == (n, 20)
    assert sliced.asym_id.shape == (n,)
    assert sliced.residue_idx.shape == (n,)
    assert sliced.atom37_coords.shape == (n, 37, 3)
    assert sliced.atom37_mask.shape == (n, 37)
    # Atom-axis: samples axis preserved, atoms trimmed.
    assert sliced.structure_coordinates.shape == (1, a, 3)
    # Bins-only fields pass through unchanged.
    assert sliced.distogram_bins.shape == (4,)
    assert sliced.pae_bins.shape == (8,)


def test_slice_padded_model_output_preserves_real_values():
    """Sliced fields must equal the corresponding ``[:n, :n]`` /
    ``[:, :a, :]`` regions of the input — the real entries, not garbage."""
    n, a = 4, 12
    b, a_bucket = 8, 32
    padded = _padded_output(b=b, a_bucket=a_bucket)

    sliced = _slice_padded_model_output(
        padded, n_real_tokens=n, n_real_atoms=a
    )

    np.testing.assert_array_equal(
        np.asarray(sliced.plddt), np.asarray(padded.plddt[:n])
    )
    np.testing.assert_array_equal(
        np.asarray(sliced.pae), np.asarray(padded.pae[:n, :n])
    )
    np.testing.assert_array_equal(
        np.asarray(sliced.distogram_logits),
        np.asarray(padded.distogram_logits[:n, :n, :]),
    )
    np.testing.assert_array_equal(
        np.asarray(sliced.structure_coordinates),
        np.asarray(padded.structure_coordinates[:, :a, :]),
    )


def test_real_shapes_from_unpadded_features_returns_none():
    """No ``token_mask`` means features were not bucket-padded — slicing is
    skipped entirely so the existing un-padded path stays a no-op."""
    features = {
        "residue_index": np.arange(20, dtype=np.int32),
        "atom_to_token_idx": np.repeat(np.arange(20, dtype=np.int32), 5),
    }
    n_real_tokens, n_real_atoms = real_shapes_from_padded_features(features)
    assert n_real_tokens is None
    assert n_real_atoms is None


def test_real_shapes_from_padded_features_recovers_pre_padding_counts():
    """For features the protenij padder built, ``token_mask.sum()`` gives the
    real token count and ``ref_mask.sum()`` gives the real atom count."""
    n, a = 5, 18
    b, a_bucket = 16, 64

    # token_mask: ``pad_features`` writes [1]*n + [0]*(b-n).
    token_mask = np.concatenate([np.ones(n), np.zeros(b - n)]).astype(np.float32)
    # atom_to_token_idx: ``pad_features`` keeps the real-atom token indices in
    # ``[0, n)`` for the first ``a`` entries, then cyclically maps padding
    # atoms to padding tokens in ``[n, b)``.
    real_to_token = np.minimum(
        np.repeat(np.arange(n, dtype=np.int32), -(-a // n))[:a],
        n - 1,
    )
    pad_to_token = (np.arange(a_bucket - a, dtype=np.int32) % (b - n)) + n
    atom_to_token_idx = np.concatenate([real_to_token, pad_to_token])

    n_real_tokens, n_real_atoms = real_shapes_from_padded_features(
        {
            "token_mask": token_mask,
            "atom_to_token_idx": atom_to_token_idx,
            "ref_mask": np.concatenate(
                [np.ones(a), np.zeros(a_bucket - a)]
            ).astype(np.float32),
        }
    )
    assert n_real_tokens == n
    assert n_real_atoms == a


def test_real_shapes_from_padded_features_handles_atom_only_padding():
    """When only the atom axis grows, padding atoms map to the final real token
    and must be counted from ``ref_mask`` instead of ``atom_to_token_idx``."""
    n = b = 5
    a, a_bucket = 18, 64

    token_mask = np.ones(b, dtype=np.float32)
    real_to_token = np.minimum(
        np.repeat(np.arange(n, dtype=np.int32), -(-a // n))[:a],
        n - 1,
    )
    pad_to_token = np.full(a_bucket - a, n - 1, dtype=np.int32)
    atom_to_token_idx = np.concatenate([real_to_token, pad_to_token])
    ref_mask = np.concatenate([np.ones(a), np.zeros(a_bucket - a)]).astype(
        np.float32
    )

    n_real_tokens, n_real_atoms = real_shapes_from_padded_features(
        {
            "token_mask": token_mask,
            "atom_to_token_idx": atom_to_token_idx,
            "ref_mask": ref_mask,
        }
    )
    assert n_real_tokens == n
    assert n_real_atoms == a


def test_atom_only_padding_shape_recovery_trims_structure_coordinates():
    n = b = 5
    a, a_bucket = 18, 64
    features = {
        "token_mask": np.ones(b, dtype=np.float32),
        "atom_to_token_idx": np.concatenate(
            [
                np.minimum(
                    np.repeat(np.arange(n, dtype=np.int32), -(-a // n))[:a],
                    n - 1,
                ),
                np.full(a_bucket - a, n - 1, dtype=np.int32),
            ]
        ),
        "ref_mask": np.concatenate([np.ones(a), np.zeros(a_bucket - a)]).astype(
            np.float32
        ),
    }

    n_real_tokens, n_real_atoms = real_shapes_from_padded_features(features)
    padded = _padded_output(b=b, a_bucket=a_bucket)
    sliced = _slice_padded_model_output(
        padded,
        n_real_tokens=n_real_tokens,
        n_real_atoms=n_real_atoms,
    )

    assert sliced.structure_coordinates.shape == (1, a, 3)
    np.testing.assert_array_equal(
        np.asarray(sliced.structure_coordinates),
        np.asarray(padded.structure_coordinates[:, :a, :]),
    )


def test_binder_target_pae_loss_is_invariant_to_padding_after_slicing():
    binder_len = 2
    n, a = 6, 18
    b, a_bucket = 12, 64
    real = _padded_output(b=n, a_bucket=a)
    padded = _embed_real_output_in_padding(
        real,
        b=b,
        a_bucket=a_bucket,
        pae_padding_value=1_000.0,
    )
    sliced = _slice_padded_model_output(
        padded,
        n_real_tokens=n,
        n_real_atoms=a,
    )

    loss = BinderTargetPAE()
    sequence = _sequence(binder_len)
    real_value, real_aux = loss(sequence=sequence, output=real, key=None)
    sliced_value, sliced_aux = loss(sequence=sequence, output=sliced, key=None)
    padded_value, _ = loss(sequence=sequence, output=padded, key=None)

    np.testing.assert_allclose(np.asarray(sliced_value), np.asarray(real_value))
    np.testing.assert_allclose(
        np.asarray(sliced_aux["bt_pae"]), np.asarray(real_aux["bt_pae"])
    )
    assert not np.isclose(float(padded_value), float(real_value))


def test_iptm_loss_is_invariant_to_padding_after_slicing():
    binder_len = 2
    n, a = 6, 18
    b, a_bucket = 24, 64
    pae_bins = jnp.arange(8, dtype=jnp.float32) + 0.5
    far_logits = jnp.full((n, n, 8), -20.0).at[..., -1].set(20.0)
    real = dataclasses.replace(
        _padded_output(b=n, a_bucket=a),
        pae_logits=far_logits,
        pae_bins=pae_bins,
    )

    low_distance_logits = jnp.full((8,), -20.0).at[0].set(20.0)
    padded_logits = jnp.full((b, b, 8), -20.0).at[..., -1].set(20.0)
    padded_logits = padded_logits.at[:, n:, :].set(low_distance_logits)
    padded_logits = padded_logits.at[n:, :, :].set(low_distance_logits)
    padded_logits = padded_logits.at[:n, :n, :].set(real.pae_logits)
    padded = _embed_real_output_in_padding(
        real,
        b=b,
        a_bucket=a_bucket,
        pae_logits=padded_logits,
    )
    sliced = _slice_padded_model_output(
        padded,
        n_real_tokens=n,
        n_real_atoms=a,
    )

    loss = IPTMLoss()
    sequence = _sequence(binder_len)
    real_value, real_aux = loss(sequence=sequence, output=real, key=None)
    sliced_value, sliced_aux = loss(sequence=sequence, output=sliced, key=None)
    padded_value, _ = loss(sequence=sequence, output=padded, key=None)

    np.testing.assert_allclose(np.asarray(sliced_value), np.asarray(real_value))
    np.testing.assert_allclose(
        np.asarray(sliced_aux["iptm"]), np.asarray(real_aux["iptm"])
    )
    assert not np.isclose(float(padded_value), float(real_value))


def test_multisample_loss_skips_slicing_when_features_unpadded():
    """Constructing ``MultiSampleProtenixLoss`` with no ``token_mask`` keeps
    ``n_real_tokens=None`` — the call path falls through to the wrapped
    ``self.loss`` with the original (unsliced) output, so existing campaigns
    are unaffected."""
    from mosaic.losses.protenix import MultiSampleProtenixLoss

    # Build with the dataclass defaults; we just want to exercise field
    # initialization, not actually run the model.
    loss = MultiSampleProtenixLoss.__new__(MultiSampleProtenixLoss)
    object.__setattr__(loss, "n_real_tokens", None)
    object.__setattr__(loss, "n_real_atoms", None)
    assert loss.n_real_tokens is None
    assert loss.n_real_atoms is None
