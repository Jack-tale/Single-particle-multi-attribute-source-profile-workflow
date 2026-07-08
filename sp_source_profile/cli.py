"""Command-line interface for sp_source_profile."""

import argparse
import sys

from . import build_source_profile


def parse_args(argv=None):
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Build single-particle source profiles from SEM-EDS data.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Example:\n"
            "  sp-build ./example_sample -o profile.xlsx --dmin 0.1 --dmax 10 --B 1000\n"
        ),
    )
    parser.add_argument(
        "--sample_path",
        type=str,
        help="Path to the sample folder (must contain element.xlsx and pics/).",
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default="source_profile.xlsx",
        help="Output Excel file path (default: source_profile.xlsx).",
    )
    parser.add_argument(
        "--dmin",
        type=float,
        default=0.1,
        help="Minimum particle diameter in um (default: 0.1).",
    )
    parser.add_argument(
        "--dmax",
        type=float,
        default=10.0,
        help="Maximum particle diameter in um (default: 10.0).",
    )
    parser.add_argument(
        "--B",
        type=int,
        default=1000,
        help="Number of bootstrap iterations (default: 1000).",
    )
    parser.add_argument(
        "--L",
        type=int,
        default=None,
        help="COBB block size. Default: auto (mean particles per field).",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.95,
        help="Confidence level for intervals (default: 0.95).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Mor_cls_model.pth",
        help="Path to the trained ResNet model checkpoint (default: Mor_cls_model.pth).",
    )
    parser.add_argument(
        "--particle-table",
        type=str,
        default=None,
        help="Separate path for the particle type table (optional).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["cpu", "cuda"],
        help="Device for model inference (default: auto-detect).",
    )
    return parser.parse_args(argv)


def main():
    """CLI entry point."""
    args = parse_args()
    try:
        output_path = build_source_profile(
            sample_path=args.sample_path,
            output=args.output,
            dmin=args.dmin,
            dmax=args.dmax,
            B=args.B,
            L=args.L,
            confidence=args.confidence,
            model_path=args.model,
            particle_table_path=args.particle_table,
            device=args.device,
        )
        print(f"\nSource profile saved to: {output_path}")
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()


"""
pip install -e .    # 安装包
sp-build \
    --sample_path /root/siton-data-zpeng/Source_profile/package_example_sample \
    --output /root/siton-data-zpeng/Source_profile/package_example_sample/package_example_profile.xlsx \
    --dmin 0.1 \
    --dmax 10


from sp_source_profile import build_source_profile
build_source_profile("example_sample", output="profile.xlsx")
"""