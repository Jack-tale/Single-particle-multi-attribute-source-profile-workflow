"""Circular Overlapping Block Bootstrap (COBB) for uncertainty quantification.

Implements the block bootstrap method specifically designed for CCSEM
single-particle data, using overlapping blocks based on the original
particle acquisition order.
"""

import math
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from .abundance import calc_abundance


def _build_blocks(df: pd.DataFrame, L: int) -> List[pd.DataFrame]:
    """Build overlapping circular blocks from the particle data.

    Parameters
    ----------
    df : pd.DataFrame
        Particle data sorted by acquisition order (Fld #).
    L : int
        Block size (number of consecutive rows per block).

    Returns
    -------
    list of pd.DataFrame
        List of overlapping blocks (circular).
    """
    n = len(df)
    step = max(1, L // 2)
    blocks = []
    for start in range(0, n, step):
        if start + L <= n:
            block = df.iloc[start : start + L]
        else:
            # Circular completion: wrap around to fill the block
            remainder = L - (n - start)
            block = pd.concat([df.iloc[start:], df.iloc[:remainder]], ignore_index=True)
        blocks.append(block)
    return blocks


def _bootstrap_one(
    blocks: List[pd.DataFrame],
    n_total: int,
    all_types: List[str],
) -> Tuple[np.ndarray, np.ndarray]:
    """Perform one bootstrap resample and return abundance vectors.

    Parameters
    ----------
    blocks : list of pd.DataFrame
        Overlapping blocks of particle data.
    n_total : int
        Number of particles to sample (original dataset size).
    all_types : list of str
        All particle type names (for consistent ordering).

    Returns
    -------
    tuple of np.ndarray
        (mass_frac_vector, num_frac_vector) aligned with all_types.
    """
    n_blocks = len(blocks)
    # Resample blocks with replacement
    selected = np.random.choice(n_blocks, size=n_blocks, replace=True)
    sampled = pd.concat([blocks[i] for i in selected], ignore_index=True)
    # Truncate to original particle count
    sampled = sampled.iloc[:n_total]
    # Compute abundance
    abd = calc_abundance(sampled)
    # Map to all_types vector
    mass_map = dict(zip(abd["Type"], abd["Mass_Frac"]))
    num_map = dict(zip(abd["Type"], abd["Num_Frac"]))
    mass_vec = np.array([mass_map.get(t, 0.0) for t in all_types])
    num_vec = np.array([num_map.get(t, 0.0) for t in all_types])
    return mass_vec, num_vec


def cobb_uncertainty(
    df: pd.DataFrame,
    B: int = 1000,
    L: Optional[int] = None,
    confidence: float = 0.95,
) -> pd.DataFrame:
    """Estimate uncertainty intervals using the COBB method.

    Parameters
    ----------
    df : pd.DataFrame
        Particle data with 'Pt_type', 'Mass' and 'Fld #' columns.
        Should already be filtered to the target size range.
    B : int
        Number of bootstrap iterations. Default 1000.
    L : int, optional
        Block size. If None, uses floor(mean particles per field).
    confidence : float
        Confidence level. Default 0.95 (for 95% CI).

    Returns
    -------
    pd.DataFrame
        Source profile with uncertainty intervals. Columns:
        Type, Mass_Frac, Mass_CI_low, Mass_CI_high,
        Num_Frac, Num_CI_low, Num_CI_high
    """
    if "Pt_type" not in df.columns or "Mass" not in df.columns or "Fld #" not in df.columns:
        raise ValueError("DataFrame must contain 'Pt_type', 'Mass', and 'Fld #' columns.")

    # Sort by Fld # to maintain acquisition order
    df_sorted = df.sort_values("Fld #").reset_index(drop=True)
    n = len(df_sorted)

    if n == 0:
        raise ValueError("Empty DataFrame after filtering.")

    # Determine block size if not specified
    if L is None:
        field_counts = df_sorted["Fld #"].value_counts()
        L = max(1, int(math.floor(field_counts.mean())))

    L = max(1, int(L))

    # Get all unique types from original data
    all_types = sorted(df_sorted["Pt_type"].unique())

    # Build overlapping blocks
    blocks = _build_blocks(df_sorted, L)

    if len(blocks) < 2:
        raise ValueError(
            f"Too few blocks ({len(blocks)}) for bootstrap. "
            "Reduce block size L or increase sample size."
        )

    # Bootstrap
    alpha = 1.0 - confidence
    mass_bootstrap = {t: [] for t in all_types}
    num_bootstrap = {t: [] for t in all_types}

    for _ in range(B):
        mass_vec, num_vec = _bootstrap_one(blocks, n, all_types)
        for i, t in enumerate(all_types):
            mass_bootstrap[t].append(mass_vec[i])
            num_bootstrap[t].append(num_vec[i])

    # Compute original abundance for point estimates
    orig_abd = calc_abundance(df_sorted)
    mass_point = dict(zip(orig_abd["Type"], orig_abd["Mass_Frac"]))
    num_point = dict(zip(orig_abd["Type"], orig_abd["Num_Frac"]))

    # Build result DataFrame
    rows = []
    for t in all_types:
        m_arr = np.array(mass_bootstrap[t])
        n_arr = np.array(num_bootstrap[t])
        ci_low = np.percentile(m_arr, alpha / 2 * 100)
        ci_high = np.percentile(m_arr, (1.0 - alpha / 2) * 100)
        ni_low = np.percentile(n_arr, alpha / 2 * 100)
        ni_high = np.percentile(n_arr, (1.0 - alpha / 2) * 100)
        rows.append({
            "Type": t,
            "Mass_Frac": mass_point.get(t, 0.0),
            "Mass_CI_low": ci_low,
            "Mass_CI_high": ci_high,
            "Num_Frac": num_point.get(t, 0.0),
            "Num_CI_low": ni_low,
            "Num_CI_high": ni_high,
        })

    result = pd.DataFrame(rows)
    # Sort by mass fraction descending
    result = result.sort_values("Mass_Frac", ascending=False).reset_index(drop=True)
    return result
