"""Cross-layer contracts for Protenij bucket padding.

These tests intentionally exercise the boundaries that slicing-only tests do
not cover: Mosaic's manual forward must agree with Protenij's canonical padded
forward, all dynamic feature leaves must have bucket-stable shapes, the public
bucket helpers must compose for valid protein sizes, and reference-conformer
masks must not be mistaken for atom-count metadata.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from mosaic.losses.protenix import (
    get_trunk_state,
    protenix_forward_from_trunk,
    real_shapes_from_padded_features,
)
from protenij.backend import load_model
from protenij.data.template import ChainInput, featurize
from protenij.padding import atom_bucket, pad_features, token_bucket


@pytest.fixture(scope="module")
def real_padded_case():
    """Load the smallest pretrained model and build one genuinely padded input."""
    model = load_model("protenix_tiny_default_v0.5.0")
    features, _, _ = featurize([ChainInput(sequence="ACDEFGHI", compute_msa=False)])
    features = jax.tree.map(
        lambda value: jnp.asarray(value) if isinstance(value, np.ndarray) else value,
        features,
    )

    n = int(features["restype"].shape[0])
    a = int(features["ref_pos"].shape[0])
    b = token_bucket(n)
    padded = pad_features(features, n=n, b=b)
    return model, padded, n, a


@pytest.mark.slow
@pytest.mark.model_forward
def test_mosaic_padded_forward_matches_real_protenij_model(real_padded_case):
    """Mosaic's split forward must equal Protenij's canonical masked forward."""
    model, padded, n, a = real_padded_case
    key = jax.random.key(0)

    expected = model(
        input_feature_dict=padded,
        N_cycle=1,
        N_sample=1,
        N_steps=1,
        key=key,
    )
    initial_embedding, trunk_state = get_trunk_state(
        model=model,
        features=padded,
        initial_recycling_state=None,
        recycling_steps=1,
        key=key,
    )
    actual = protenix_forward_from_trunk(
        model=model,
        features=padded,
        initial_embedding=initial_embedding,
        trunk_state=trunk_state,
        sampling_steps=1,
        key=key,
    )

    np.testing.assert_allclose(
        np.asarray(actual.distogram_logits[:n, :n]),
        np.asarray(expected.distogram_logits[:n, :n]),
        rtol=1e-3,
        atol=1e-3,
    )
    np.testing.assert_allclose(
        np.asarray(actual.structure_coordinates[0, :a]),
        np.asarray(expected.coordinates[0, :a]),
        rtol=1e-2,
        atol=5e-2,
    )
    np.testing.assert_allclose(
        np.asarray(actual.pae_logits[:n, :n]),
        np.asarray(expected.confidence_metrics.pae_logits[0, :n, :n]),
        rtol=5e-3,
        atol=2e-2,
    )


def _bucketable_features(n: int, a: int) -> dict:
    return {
        "restype": jnp.zeros((n, 32), dtype=jnp.float32),
        "ref_pos": jnp.zeros((a, 3), dtype=jnp.float32),
        "ref_mask": jnp.ones((a,), dtype=jnp.float32),
        "atom_to_token_idx": jnp.arange(a, dtype=jnp.int32) % n,
        "constraint_feature": {
            "contact": jnp.zeros((n, n, 2), dtype=jnp.float32),
            "pocket": jnp.zeros((n, n, 1), dtype=jnp.float32),
        },
    }


def test_pad_features_pads_nested_constraint_pair_axes():
    n, a, b, a_bucket = 10, 50, 32, 128
    padded = pad_features(
        _bucketable_features(n, a),
        n=n,
        b=b,
        a_bucket=a_bucket,
    )

    assert padded["constraint_feature"]["contact"].shape == (b, b, 2)
    assert padded["constraint_feature"]["pocket"].shape == (b, b, 1)


def test_different_lengths_in_one_bucket_share_a_jit_trace():
    b, a_bucket = 32, 128
    padded_10 = pad_features(
        _bucketable_features(10, 50),
        n=10,
        b=b,
        a_bucket=a_bucket,
    )
    padded_11 = pad_features(
        _bucketable_features(11, 55),
        n=11,
        b=b,
        a_bucket=a_bucket,
    )
    traces: list[tuple[tuple[int, ...], ...]] = []

    def _sum_restype(features):
        traces.append(tuple(leaf.shape for leaf in jax.tree.leaves(features)))
        return features["restype"].sum()

    compiled = eqx.filter_jit(_sum_restype)
    jax.block_until_ready(compiled(padded_10))
    jax.block_until_ready(compiled(padded_11))

    assert len(traces) == 1, f"bucketed inputs retraced with shapes {traces}"


def test_pad_features_automatically_includes_padding_token_atom_slack():
    """33 valines have 232 heavy atoms, leaving too little slack in A=256."""
    n, a = 33, 232
    b = token_bucket(n)
    expected_a_bucket = atom_bucket(a + (b - n))

    padded = pad_features(
        {"ref_pos": jnp.zeros((a, 3), dtype=jnp.float32)},
        n=n,
        b=b,
    )

    assert padded["token_mask"].shape == (b,)
    assert padded["ref_pos"].shape == (expected_a_bucket, 3)


def test_real_atom_count_does_not_use_reference_conformer_mask_sum():
    n, a, b, a_bucket = 2, 5, 4, 8
    features = {
        "ref_pos": jnp.zeros((a, 3), dtype=jnp.float32),
        # Atom 1 is real but lacks a valid reference conformer.
        "ref_mask": jnp.array([1.0, 0.0, 1.0, 1.0, 1.0], dtype=jnp.float32),
        "atom_to_token_idx": jnp.array([0, 0, 1, 1, 1], dtype=jnp.int32),
    }
    padded = pad_features(
        features,
        n=n,
        b=b,
        a_bucket=a_bucket,
    )

    n_real_tokens, n_real_atoms = real_shapes_from_padded_features(padded)

    assert n_real_tokens == n
    assert n_real_atoms == a
