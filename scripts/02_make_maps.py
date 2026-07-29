from fermipy.gtanalysis import GTAnalysis
from astropy.io import fits
from astropy.wcs import WCS
import matplotlib.pyplot as plt
import numpy as np

CONFIG = "/Users/maomao/sfsu_research/minniti/configs/minniti_standard_config.yaml"
BASELINE_NAME = "baseline_standard"

print("=" * 70)
print("Loading baseline ROI...")
print("=" * 70)

gta = GTAnalysis(CONFIG, logging={"verbosity": 3})
gta.setup()

# Load fitted background model from baseline fit
gta.load_roi(BASELINE_NAME)

# Check coordinate access
src = gta.roi.point_sources[0]

print(type(src))
print(src)

print("\nAttributes:")
print([a for a in dir(src) if not a.startswith("_")])

print("\nDictionary keys (if available):")
try:
    print(src.keys())
except Exception as e:
    print(e)

print("\nRA:", getattr(src, "ra", None))
print("DEC:", getattr(src, "dec", None))
print("GLON:", getattr(src, "glon", None))
print("GLAT:", getattr(src, "glat", None))
print("TS:", getattr(src, "ts", None))

# Create maps
print("\nCreating residual map...")
resid = gta.residmap("baseline_residual",model={},make_plots=False)
print("\nResidual map complete.")

print("\nCreating TS map...")
tsmap = gta.tsmap(
    "baseline_tsmap",
    model={
        "SpatialModel": "PointSource",
        "SpectrumType": "PowerLaw",
        "Index": 2.0
    },
    make_plots=False
)
print("\nTS map complete.")

print("\nMaps saved by Fermipy.\n")

print("=" * 70)
print("Opening FITS files...")
print("=" * 70)

# -------------------------
# Residual Map
# -------------------------

resid_hdul = fits.open("/Users/maomao/sfsu_research/minniti/results/standard_diffuse/baseline/baseline_residual_pointsource_powerlaw_2.00_residmap.fits")

print("\nResidual FITS Extensions:")
resid_hdul.info()

# Usually image is in extension 0 or 1
for hdu in resid_hdul:
    if hdu.data is not None:
        resid_data = hdu.data
        break

plt.figure(figsize=(8,7))
plt.imshow(resid_data, origin="lower")
plt.colorbar(label="Residual")
plt.title("Residual Map")
plt.xlabel("Pixel")
plt.ylabel("Pixel")

# -------------------------
# TS Map
# -------------------------
ts_hdul = fits.open("/Users/maomao/sfsu_research/minniti/results/standard_diffuse/baseline/baseline_tsmap_pointsource_powerlaw_2.00_tsmap.fits")

# Get sqrt(TS) image
if "SQRT_TS_MAP" in ts_hdul:
    hdu = ts_hdul["SQRT_TS_MAP"]
elif "TS_MAP" in ts_hdul:
    hdu = ts_hdul["TS_MAP"]
else:
    hdu = ts_hdul[0]

ts_data = hdu.data
wcs = WCS(hdu.header)

fig = plt.figure(figsize=(10,9))
ax = fig.add_subplot(111, projection=wcs)
im = ax.imshow(ts_data,origin="lower",cmap="magma",vmin=0,vmax=np.nanpercentile(ts_data,99.5))
cbar = plt.colorbar(im, ax=ax)
cbar.set_label(r"$\sqrt{\rm TS}$", fontsize=14)
ax.coords[0].set_axislabel("Galactic Longitude")
ax.coords[1].set_axislabel("Galactic Latitude")
ax.grid(color="white", alpha=0.35)
ax.set_title("sqrt(TS) Map")

TS_LABEL_THRESHOLD = 25

for src in gta.roi.sources:
    try:
        ts = src["ts"]
    except Exception:
        continue

    if ts < TS_LABEL_THRESHOLD:
        continue

    sky = src.skydir

    ax.plot_coord(sky,marker="+",color="white",ms=8)

    ax.text_coord(sky,src.name,color="white",fontsize=7)

plt.tight_layout()
plt.show()

print("\nFinished.")