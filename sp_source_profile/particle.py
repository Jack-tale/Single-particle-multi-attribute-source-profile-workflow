"""Data loading, particle type assignment, and size filtering."""

import os
from typing import Dict, List, Optional

import pandas as pd

from .chemistry import NON_O_ELEMENTS, get_chemical_label


def read_element_data(sample_path: str) -> pd.DataFrame:
    """Read the element.xlsx file from a sample directory.

    Parameters
    ----------
    sample_path : str
        Path to the sample folder containing element.xlsx.

    Returns
    -------
    pd.DataFrame
        DataFrame with particle data.
    """
    xlsx_path = os.path.join(sample_path, "element.xlsx")
    if not os.path.isfile(xlsx_path):
        raise FileNotFoundError(f"element.xlsx not found in {sample_path}")
    df = pd.read_excel(xlsx_path)
    # Ensure numeric types for key columns
    for col in ["Part #", "Fld #", "Mass", "Davg"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def assign_type(
    df: pd.DataFrame,
    morph_dict: Dict[int, str],
    element_cols: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Assign coupled type labels to each particle.

    The coupled type format is: "ChemLabel (MorphLabel)"

    Parameters
    ----------
    df : pd.DataFrame
        Particle data containing element columns and Part #.
    morph_dict : dict of {int: str}
        Mapping from Part # to morphology label.
    element_cols : list of str, optional
        Element columns for chemical classification.

    Returns
    -------
    pd.DataFrame
        DataFrame with added 'Pt_type' column.
    """
    if element_cols is None:
        element_cols = NON_O_ELEMENTS

    df = df.copy()

    def _assign(row):
        pid = int(row["Part #"])
        morph = morph_dict.get(pid, "Unclassified")
        chem = get_chemical_label(row, element_cols)
        return f"{chem} ({morph})"

    df["Pt_type"] = df.apply(_assign, axis=1)
    return df


def filter_by_size(
    df: pd.DataFrame,
    dmin: float = 0.1,
    dmax: float = 10.0,
) -> pd.DataFrame:
    """Filter particles by equivalent spherical diameter.

    Parameters
    ----------
    df : pd.DataFrame
        Particle data with 'Davg' column.
    dmin : float
        Minimum diameter (um). Default 0.1.
    dmax : float
        Maximum diameter (um). Default 10.0.

    Returns
    -------
    pd.DataFrame
        Filtered DataFrame.
    """
    if "Davg" not in df.columns:
        raise ValueError("DataFrame must contain 'Davg' column for size filtering.")
    return df[(df["Davg"] >= dmin) & (df["Davg"] <= dmax)].copy()


def export_particle_table(
    df: pd.DataFrame,
    output_path: str,
    sheet_name: str = "Particle Types",
) -> str:
    """Export the particle type table to an Excel sheet.

    Columns exported: Part #, Fld #, Mass, Davg, Pt_type

    Parameters
    ----------
    df : pd.DataFrame
        Particle data with Pt_type column.
    output_path : str
        Path to the output Excel file.
    sheet_name : str
        Sheet name in the Excel file.

    Returns
    -------
    str
        Path to the saved file.
    """
    cols = ["Part #", "Fld #", "Mass", "Davg", "Pt_type"]
    existing = [c for c in cols if c in df.columns]
    out_df = df[existing].copy()
    out_df.to_excel(output_path, sheet_name=sheet_name, index=False)
    return output_path
