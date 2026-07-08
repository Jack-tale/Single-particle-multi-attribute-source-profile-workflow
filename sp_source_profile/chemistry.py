"""Elemental composition classification for single particles.

Based on EDS data, assigns a chemical label to each particle using
the top non-O elements (excluding O).
"""

NON_O_ELEMENTS = [
    "C", "Al", "Si", "Ca", "Fe", "S", "Sn", "Ti", "Mn", "Ni", "Cu",
    "Pb", "Na", "Mg", "K", "Cl", "Zn", "Ba", "V", "Cr", "Co", "P", "Se",
]


def get_chemical_label(row, element_cols=None, rest_C=True):
    """Assign a chemical label based on top non-O elements.

    Rules:
    - Exclude O.
    - Temporarily set C to 0; if all other non-O elements are 0,
      set C to 100 to ensure pure carbon particles get label 'C-O'.
    - renormalize remaining elements to sum to 100%.
    - If 0 non-O elements -> 'Unk'
    - If 1 non-O element (X) -> 'X-O'
    - If 2 non-O elements (X, Y) sorted by abundance -> 'X-Y'
    - If 3+ -> take top 3 -> 'X-Y-Z'

    Parameters
    ----------
    row : pd.Series
        A row from the element DataFrame containing element columns.
    element_cols : list of str, optional
        Columns to use for classification. Defaults to NON_O_ELEMENTS.

    Returns
    -------
    str
        Chemical label string.
    """
    if element_cols is None:
        element_cols = NON_O_ELEMENTS

    raw_values = {}
    for el in element_cols:
        v = row.get(el, 0)
        if v is None:
            v = 0
        raw_values[el] = float(v)

    # Apply special C rule: set C to 0 first
    values = raw_values.copy()

    if "C" in values and rest_C:
        values["C"] = 0.0

        # If all other elements are zero, treat particle as pure C (label C-O)
        other_nonzero = any(val > 0 for el, val in values.items() if el != "C")
        if not other_nonzero:
            values["C"] = 100.0

    present = {el: val for el, val in values.items() if val > 0}

    if not present:
        return "Unk"

    total = sum(present.values())
    if total <= 0:
        return "Unk"

    sorted_els = sorted(present.keys(), key=lambda el: present[el] / total, reverse=True)

    n = len(sorted_els)
    if n == 1:
        return f"{sorted_els[0]}-O"
    elif n == 2:
        return f"{sorted_els[0]}-{sorted_els[1]}"
    else:
        return f"{sorted_els[0]}-{sorted_els[1]}-{sorted_els[2]}"
