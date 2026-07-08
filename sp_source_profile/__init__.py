import os
from typing import Optional
import pandas as pd
import torch

from .morphology import load_model, get_class_names, classify_images
from .particle import read_element_data, assign_type, filter_by_size, export_particle_table
from .abundance import calc_abundance
from .cobb import cobb_uncertainty


def build_source_profile(
    sample_path: str,
    output: str = "source_profile.xlsx",
    dmin: float = 0.1,
    dmax: float = 10.0,
    B: int = 1000,
    L: Optional[int] = None,
    confidence: float = 0.95,
    model_path: str = "Mor_cls_model.pth",
    particle_table_path: Optional[str] = None,
    device: Optional[str] = None,
) -> str:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"[1/6] Loading element data from {sample_path}")
    df = read_element_data(sample_path)
    total_particles = len(df)
    print(f"       Loaded {total_particles} particles")

    pics_dir = os.path.join(sample_path, "pics")
    if not os.path.isdir(pics_dir):
        raise FileNotFoundError(f"pics/ directory not found in {sample_path}")

    print(f"[2/6] Loading morphology model from {model_path}")
    model_device = torch.device(device)
    model = load_model(model_path, model_device)
    class_names = get_class_names(model_path)
    print(f"       Using classes: {class_names}")

    print(f"[3/6] Classifying particle morphologies")
    part_numbers = df["Part #"].astype(int).tolist()
    morph_dict = classify_images(model, model_device, pics_dir, part_numbers, class_names)
    n_classified = sum(1 for v in morph_dict.values() if v != "Unclassified")
    print(f"       Classified {n_classified}/{total_particles} particles")

    print(f"[4/6] Assigning particle types")
    df = assign_type(df, morph_dict)

    if particle_table_path:
        print(f"       Exporting particle type table to {particle_table_path}")
        export_particle_table(df, particle_table_path)

    print(f"[5/6] Filtering by size range [{dmin}, {dmax}] um")
    df_filtered = filter_by_size(df, dmin, dmax)
    n_filtered = len(df_filtered)
    print(f"       {n_filtered}/{total_particles} particles remain after filtering")

    cols = ["Part #", "Fld #", "Mass", "Davg", "Pt_type"]
    particle_cols = [c for c in cols if c in df_filtered.columns]

    print(f"[6/6] Computing abundance and COBB uncertainty (B={B})")
    result = cobb_uncertainty(df_filtered, B=B, L=L, confidence=confidence)

    print(f"       Writing results to {output}")
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df_filtered[particle_cols].to_excel(
            writer, sheet_name="Particle Types", index=False
        )
        result.to_excel(writer, sheet_name="Source Profile", index=False)

    print("Done!")
    return output
