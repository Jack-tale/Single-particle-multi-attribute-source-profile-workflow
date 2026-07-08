"""Abundance calculation for single particle source profiles."""

import pandas as pd


def calc_abundance(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate mass and number fractions for each particle type.

    Parameters
    ----------
    df : pd.DataFrame
        Particle data with 'Pt_type' and 'Mass' columns.

    Returns
    -------
    pd.DataFrame
        Abundance table with columns:
        Type, Mass_Frac, Num_Frac
    """
    if "Pt_type" not in df.columns or "Mass" not in df.columns:
        raise ValueError("DataFrame must contain 'Pt_type' and 'Mass' columns.")

    total_mass = df["Mass"].sum()
    total_count = len(df)

    if total_mass <= 0 or total_count <= 0:
        raise ValueError("Total mass or count is zero; cannot compute abundance.")

    grouped = df.groupby("Pt_type")
    mass_frac = grouped["Mass"].sum() / total_mass
    num_frac = grouped.size() / total_count

    result = pd.DataFrame({
        "Type": mass_frac.index,
        "Mass_Frac": mass_frac.values,
        "Num_Frac": num_frac.values,
    }).reset_index(drop=True)

    # Sort by mass fraction descending
    result = result.sort_values("Mass_Frac", ascending=False).reset_index(drop=True)
    return result
