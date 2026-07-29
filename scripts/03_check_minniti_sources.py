import argparse
import csv
import math
import os
import traceback

from astropy.coordinates import SkyCoord
import astropy.units as u
from fermipy.gtanalysis import GTAnalysis


CONFIG = "/Users/maomao/sfsu_research/minniti/configs/minniti_standard_config.yaml"
TARGETS = "/Users/maomao/sfsu_research/minniti/data/minniti_catalog.csv"
BASELINE_NAME = "baseline_standard"

BASELINE_OUTDIR = "/Users/maomao/sfsu_research/minniti/results/standard_diffuse/baseline"
PER_TARGET_DIR = "/Users/maomao/sfsu_research/minniti/results/standard_diffuse/per_target"
PER_TARGET_TABLE_DIR = "/Users/maomao/sfsu_research/minniti/results/standard_diffuse/tables/per_target"


def read_targets(path):
    targets = {}
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            targets[row["name"]] = {
                "name": row["name"],
                "glon": float(row["glon"]),
                "glat": float(row["glat"]),
            }
    return targets


def fix_all_sources(gta):
    for src in gta.roi.sources:
        gta.free_source(src.name, free=False)


def free_local_background(gta, glon, glat, radius=3.0, exclude=None):
    """
    Free diffuse components and nearby catalog source normalizations around one target.
    """
    if exclude is None:
        exclude = []

    fix_all_sources(gta)

    gta.free_source("galdiff")
    gta.free_source("isodiff")

    skydir = SkyCoord(l=glon * u.deg, b=glat * u.deg, frame="galactic")

    gta.free_sources(
        skydir=skydir,
        distance=radius,
        pars="norm",
        exclude=["galdiff", "isodiff"] + exclude,
    )


def add_minniti_source(gta, name, glon, glat):
    gta.add_source(
        name,
        {
            "glon": glon,
            "glat": glat,
            "SpatialModel": "PointSource",
            "SpectrumType": "PowerLaw",
            "Index": 2.0,
            "Scale": 1000,
            "Prefactor": 1e-13,
        },
        free=True,
        save_source_maps=False,
    )

    # First-pass broadband scan: free normalization only.
    gta.free_source(name, pars="norm")


def src_value(src, key):
    try:
        val = src[key]
        try:
            return float(val)
        except Exception:
            return val
    except Exception:
        return ""


def write_one_row(path, row):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    fieldnames = [
        "name",
        "glon",
        "glat",
        "loglike_null",
        "loglike_alt",
        "TS_manual",
        "sqrt_TS",
        "fit_quality_null",
        "fit_status_null",
        "fit_quality_alt",
        "fit_status_alt",
        "fermipy_ts",
        "flux",
        "flux_err",
        "eflux",
        "eflux_err",
        "npred",
        "error",
    ]

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(row)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("target_name", help="Example: Minniti14")
    parser.add_argument("--radius", type=float, default=3.0)
    args = parser.parse_args()

    target_name = args.target_name

    targets = read_targets(TARGETS)

    if target_name not in targets:
        raise ValueError(f"Target {target_name} not found in {TARGETS}")

    target = targets[target_name]
    name = target["name"]
    glon = target["glon"]
    glat = target["glat"]

    os.makedirs(PER_TARGET_DIR, exist_ok=True)
    os.makedirs(PER_TARGET_TABLE_DIR, exist_ok=True)

    target_outdir = os.path.join(PER_TARGET_DIR, name)
    os.makedirs(target_outdir, exist_ok=True)

    target_log = os.path.join(target_outdir, f"{name}.log")

    print("\n" + "=" * 80)
    print(f"Testing {name}: l={glon:.3f}, b={glat:.3f}")
    print("=" * 80)

    try:
        # Use same baseline outdir/cache from script 1.
        # Unique logfile prevents all parallel jobs fighting over fermipy.log.
        gta = GTAnalysis(
            CONFIG,
            logging={"verbosity": 3},
            fileio={
                "outdir": BASELINE_OUTDIR,
                "logfile": target_log,
            },
        )

        gta.setup()

        # -------------------------
        # Null model: no Minniti source
        # -------------------------
        gta.load_roi(BASELINE_NAME)

        free_local_background(gta, glon, glat, radius=args.radius)

        print("Running null fit...")
        fit_null = gta.fit()
        loglike_null = float(fit_null["loglike"])

        # -------------------------
        # Alternative model: add Minniti source
        # -------------------------
        gta.load_roi(BASELINE_NAME)

        add_minniti_source(gta, name, glon, glat)

        free_local_background(gta, glon, glat, radius=args.radius, exclude=[name])
        gta.free_source(name, pars="norm")

        print("Running alternative fit...")
        fit_alt = gta.fit()
        gta.print_model()
        loglike_alt = float(fit_alt["loglike"])

        ts = 2.0 * (loglike_alt - loglike_null)
        if ts < 0:
            ts = 0.0

        src = gta.roi[name]

        roi_prefix = os.path.join("..", "per_target", name, f"{name}_fit")
        gta.write_roi(roi_prefix)

        row = {
            "name": name,
            "glon": glon,
            "glat": glat,
            "loglike_null": loglike_null,
            "loglike_alt": loglike_alt,
            "TS_manual": ts,
            "sqrt_TS": math.sqrt(ts),
            "fit_quality_null": fit_null.get("fit_quality", ""),
            "fit_status_null": fit_null.get("fit_status", ""),
            "fit_quality_alt": fit_alt.get("fit_quality", ""),
            "fit_status_alt": fit_alt.get("fit_status", ""),
            "fermipy_ts": src_value(src, "ts"),
            "flux": src_value(src, "flux"),
            "flux_err": src_value(src, "flux_err"),
            "eflux": src_value(src, "eflux"),
            "eflux_err": src_value(src, "eflux_err"),
            "npred": src_value(src, "npred"),
            "error": "",
        }

        print(f"{name}: TS_manual={ts:.3f}, sqrt_TS={math.sqrt(ts):.3f}")

    except Exception as e:
        traceback.print_exc()

        row = {
            "name": name,
            "glon": glon,
            "glat": glat,
            "loglike_null": "",
            "loglike_alt": "",
            "TS_manual": "",
            "sqrt_TS": "",
            "fit_quality_null": "",
            "fit_status_null": "",
            "fit_quality_alt": "",
            "fit_status_alt": "",
            "fermipy_ts": "",
            "flux": "",
            "flux_err": "",
            "eflux": "",
            "eflux_err": "",
            "npred": "",
            "error": str(e),
        }

    out_csv = os.path.join(PER_TARGET_TABLE_DIR, f"{name}.csv")
    write_one_row(out_csv, row)

    print(f"Saved target result to: {out_csv}")


if __name__ == "__main__":
    main()
