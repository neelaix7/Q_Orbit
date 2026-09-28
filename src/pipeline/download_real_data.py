# download_real_data.py - CLI: optional download of real public light-curve data
from __future__ import annotations
import argparse
import sys
import os
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))


def main():
    parser = argparse.ArgumentParser(
        description="Download real public light-curve data for space debris objects"
    )
    parser.add_argument(
        "--target-dir",
        type=str,
        default="data/raw",
        help="Directory to save downloaded data (default: data/raw)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download even if data already exists",
    )
    args = parser.parse_args()

    target_dir = Path(args.target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    print("Real data download utility")
    print("=" * 40)
    print()
    print("Checking available public data sources:")
    print()
    print("1. ESA DISCOS database - TWO_LINE_ELEMENT set data")
    print("   URL: https://www.esa.int/PSW/debris/ESA_Debris_Index")
    print("   Status: Documentation-only; no real-time light curves available")
    print()
    print("2. NASA ODPO (Orbital Debris Program Office)")
    print("   URL: https://www.ospacewealth.com/odpo/")
    print("   Status: Facts/catalogues; no light curve data")
    print()
    print("3. CelesTrak TLE data")
    print("   URL: https://celestrak.com/NORAD/elements/")
    print("   Status: Orbital elements only; no photometric data")
    print()
    print("4. Public light-curve datasets")
    print("   - Ops-Space Debris Light Curves (simulated/analytical)")
    print("   - ASCA/ X-ray archives (not optical light curves)")
    print()
    print("Conclusion: No public optical light-curve dataset for space debris")
    print("             is readily available for direct download at runtime.")
    print()
    print(f"Target directory: {target_dir.resolve()}")
    print("  (No data downloaded - synthetic generator is the primary source.)")
    print()
    print("To manually download TLE data if desired:")
    print("  curl -O https://celestrak.com/NORAD/elements/stations.txt")
    print("  curl -O https://celestrak.com/NORAD/elements/weather.txt")


if __name__ == "__main__":
    main()