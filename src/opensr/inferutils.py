"""Sentinel-2 10 m -> 1 m RGB super-resolution for any set of points and dates.

10 m -> 2.5 m is the SEN2-SR model (x4); 2.5 m -> 1 m is bicubic resampling plus
sharpening. The 1 m grid is finer, but detail below ~2.5 m is not recovered.

    import opensr.inferutils
    results = opensr.inferutils.run([(105.392986, 14.121299)], ["2026-10-29", "2026-11-08"])
"""
from __future__ import annotations

import numbers
import os
import re
import threading
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from itertools import product

# GDAL reads these when a file is opened; setdefault keeps any value the user already set.
for _k, _v in dict(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", GDAL_HTTP_MULTIPLEX="YES",
                   GDAL_HTTP_VERSION="2", GDAL_HTTP_MAX_RETRY="5",
                   GDAL_HTTP_RETRY_DELAY="2").items():
    os.environ.setdefault(_k, _v)

import mlstac
import numpy as np
import planetary_computer
import rasterio
import torch
import torch.nn.functional as F
from opensr_srgan import load_inference_model
from opensr_srgan.utils.back_projection import BackProjectionConfig, back_projection
from PIL import Image
from pystac_client import Client
from rasterio.enums import Resampling
from rasterio.transform import from_origin
from rasterio.warp import transform as warp_transform
from rasterio.windows import from_bounds
from shapely.geometry import Point, shape

PATCH = 512              # 10 m px per side = 5.12 km box
SEARCH_DAYS = 10         # no pass on the exact date -> nearest within +/- this
STRETCH = 0.3            # reflectance shown as white in RGB
SCALE = 4                # model: 10 m -> 2.5 m
OUT_RES = 1.0            # final pixel size (m); 2.5 m -> 1 m by resampling
TILE, OVERLAP = 128, 32  # model tile (10 m px), feathered overlap
BANDS = ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"]
RGB = [2, 1, 0]          # B04, B03, B02
STAC = "https://planetarycomputer.microsoft.com/api/stac/v1"
WEIGHTS_DIR = os.path.expanduser("~/.cache/opensr/SEN2SRLite")
WEIGHTS_URL = "https://huggingface.co/tacofoundation/sen2sr/resolve/main/SEN2SRLite/main/mlm.json"
DEV = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class Result:
    index: int                          # 1-based location index
    lon: float
    lat: float
    date: str                           # requested date (the scene may be up to SEARCH_DAYS away)
    path: str | None = None             # 1 m RGB GeoTIFF; None if the job failed
    preview: Image.Image | None = field(default=None, repr=False)  # centre 1 km, 10 m input | 1 m output
    error: str | None = None


# ---------------- Inputs ----------------
def _parse_line(line):
    try:
        return tuple(float(v) for v in re.split(r"[\s,;]+", line.strip())[:2])
    except ValueError:  # header / blank line
        return None


def _points(lonlat):
    """(lon, lat) | [(lon, lat), ...] | CSV text or CSV file path -> [(lon, lat), ...]"""
    if isinstance(lonlat, str):
        text = open(lonlat).read() if os.path.isfile(lonlat) else lonlat
        pts = [p for p in map(_parse_line, text.splitlines()) if p and len(p) == 2]
    elif isinstance(lonlat[0], numbers.Real):
        pts = [lonlat]
    else:
        pts = list(lonlat)
    if not pts:
        raise ValueError("no lon/lat points given")
    out = []
    for lon, lat in pts:
        lon, lat = float(lon), float(lat)
        if not (-180 <= lon <= 180 and -90 <= lat <= 90):
            raise ValueError(f"({lon}, {lat}) is not a valid (lon, lat); lon comes first")
        out.append((lon, lat))
    return out


def _dates(date):
    """'YYYY-MM-DD' | datetime.date | list of either -> ['YYYY-MM-DD', ...]"""
    dates = [str(d)[:10] for d in (date if isinstance(date, (list, tuple)) else [date])]
    for d in dates:
        datetime.strptime(d, "%Y-%m-%d")  # fail before any download, not inside a worker
    return dates


def _locations(location, n):
    """None | name | list of n names (None allowed) -> n file-safe names or None."""
    names = location if isinstance(location, (list, tuple)) else [location] * n
    if len(names) != n:
        raise ValueError(f"got {len(names)} location names for {n} points")
    # Keep any script (e.g. Khmer); only drop whitespace and characters paths can't hold.
    return [re.sub(r'[\s<>:"/\\|?*\x00-\x1f]+', "_", str(s)).strip("_.") or None if s else None
            for s in names]


def _file_stems(jobs, names):
    """output_<date>_<name>, then (1), (2)... on repeats; output_<date>_run_<n> if unnamed.

    Takes the first free candidate, compared case-insensitively (macOS/Windows), so a
    user name like "run_1" or "Site(1)" can never overwrite another job's file.
    """
    used, stems = set(), []
    for r in jobs:
        name = names[r.index - 1]
        base = f"output_{r.date}_{name or 'run'}"
        suffix, k = ("({})", 0) if name else ("_{}", 1)
        stem = base + suffix.format(k) if k else base
        while stem.lower() in used:
            k += 1
            stem = base + suffix.format(k)
        used.add(stem.lower())
        stems.append(stem)
    return stems


# ---------------- Model ----------------
class _ESRGAN_RGBN(torch.nn.Module):
    """SEN2-SR ESRGAN (EMA weights, expects 0-1 reflectance) + back-projection."""
    def __init__(self, g):
        super().__init__()
        self.g = g

    def forward(self, x):
        return back_projection(self.g(x), x, BackProjectionConfig())


_model = None
_gpu = threading.Lock()   # one job on the GPU at a time; downloads run in parallel
_batch = 36               # tiles per forward pass; halved on out-of-memory


def _load_model():
    """Load once per process, on the first run() call (not at import)."""
    global _model
    if _model is None:
        torch.backends.cudnn.benchmark = True
        if not os.path.exists(f"{WEIGHTS_DIR}/mlm.json"):
            mlstac.download(file=WEIGHTS_URL, output_dir=WEIGHTS_DIR)
        m = mlstac.load(WEIGHTS_DIR).compiled_model(device=DEV)
        m.sr_model = _ESRGAN_RGBN(load_inference_model("SEN2-SR", map_location="cpu"))
        _model = m.to(DEV).eval()
    return _model


# ---------------- GPU processing ----------------
def _super_resolve(x, batch):
    """(C,H,W) 10 m -> (C,4H,4W) 2.5 m, `batch` tiles per pass, feathered seams."""
    C, H, W = x.shape
    step, T = TILE - OVERLAP, TILE * SCALE
    ys = sorted({min(y, H - TILE) for y in range(0, H - OVERLAP, step)})
    xs = sorted({min(c, W - TILE) for c in range(0, W - OVERLAP, step)})
    i = torch.arange(T, device=x.device) + 0.5
    w = (torch.minimum(i, T - i) / (OVERLAP * SCALE)).clamp(max=1)
    w = (w[:, None] * w[None])[None]
    out = torch.zeros(C, H * SCALE, W * SCALE, device=x.device)
    wsum = torch.zeros(1, H * SCALE, W * SCALE, device=x.device)
    coords = list(product(ys, xs))
    for k in range(0, len(coords), batch):
        cs = coords[k:k + batch]
        for p, (y, c) in zip(_model(torch.stack([x[:, y:y + TILE, c:c + TILE] for y, c in cs])), cs):
            out[:, y * SCALE:y * SCALE + T, c * SCALE:c * SCALE + T] += p * w
            wsum[:, y * SCALE:y * SCALE + T, c * SCALE:c * SCALE + T] += w
    return out.div_(wsum).clamp_(0, 1)


def _unsharp(img, sigma=0.8, amount=0.25):
    r = int(3 * sigma + 0.5)
    k = torch.exp(-torch.arange(-r, r + 1, device=img.device, dtype=img.dtype) ** 2 / (2 * sigma ** 2))
    k /= k.sum()
    b = F.pad(img[:, None], (r, r, r, r), mode="reflect")
    b = F.conv2d(F.conv2d(b, k.view(1, 1, 1, -1)), k.view(1, 1, -1, 1))[:, 0]
    return img + amount * (img - b)


def _to_uint8(rgb):
    return ((rgb / STRETCH).clamp(0, 1) * 255).round().to(torch.uint8)


def _gpu_products(lr):
    """10 m numpy (10,H,W) -> uint8 1 m RGB (3, 10H/OUT_RES, 10W/OUT_RES) numpy."""
    global _batch
    x = torch.from_numpy(lr).to(DEV)
    with _gpu, torch.inference_mode():
        while True:
            try:
                sr = _super_resolve(x, _batch)
                break
            except torch.cuda.OutOfMemoryError:
                if _batch == 1:
                    raise
                _batch //= 2
                torch.cuda.empty_cache()
        rgb = _unsharp(sr[RGB])  # sharpen at the model's native 2.5 m
        size = (round(lr.shape[1] * 10 / OUT_RES), round(lr.shape[2] * 10 / OUT_RES))
        rgb = F.interpolate(rgb[None], size=size, mode="bicubic", align_corners=False)[0]
        return _to_uint8(rgb).cpu().numpy()


# ---------------- Sentinel-2 access ----------------
def _search_date(date, points):
    """One STAC search per date covering every location."""
    d0 = datetime.strptime(date, "%Y-%m-%d").date()
    lons, lats = zip(*points)
    items = Client.open(STAC).search(
        collections=["sentinel-2-l2a"],
        bbox=[min(lons), min(lats), max(lons), max(lats)],
        datetime=f"{d0 - timedelta(SEARCH_DAYS)}/{d0 + timedelta(SEARCH_DAYS)}",
    ).item_collection()
    return d0, [(i, shape(i.geometry)) for i in items]


def _find_scenes(lon, lat, scenes):
    """Scenes covering the point: nearest date first, then least cloudy."""
    d0, items = scenes
    hits = [i for i, fp in items if fp.contains(Point(lon, lat))]
    return sorted(hits, key=lambda i: (abs((i.datetime.date() - d0).days),
                                       i.properties.get("eo:cloud_cover", 100)))


def _open_asset(item, key):  # sign at read time so tokens refresh on long runs
    return rasterio.open(planetary_computer.sign(item.assets[key].href))


def _patch_geo(item, lon, lat):
    with _open_asset(item, "B04") as ds:
        (x,), (y,) = warp_transform("EPSG:4326", ds.crs, [lon], [lat])
        t, half = ds.transform, PATCH // 2 * 10
        return ds.crs, t.c + (x - t.c) // 10 * 10 - half, t.f - (t.f - y) // 10 * 10 + half


def _read_window(item, key, left, top, resampling):
    with _open_asset(item, key) as ds:
        win = from_bounds(left, top - PATCH * 10, left + PATCH * 10, top, ds.transform)
        return ds.read(1, window=win, out_shape=(PATCH, PATCH), boundless=True,
                       fill_value=0, resampling=resampling)


def _read_band(item, band, left, top):
    dn = _read_window(item, band, left, top, Resampling.bilinear).astype("float32")
    rb = item.assets[band].extra_fields.get("raster:bands", [{}])[0]
    new_baseline = item.properties.get("s2:processing_baseline", "00.00") >= "04.00"
    offset = rb.get("offset", -0.1 if new_baseline else 0.0)
    return np.clip(np.where(dn > 0, dn * rb.get("scale", 1e-4) + offset, 0.0), 0, 1)


# ---------------- One job ----------------
def _center(chw, half):
    cy, cx = chw.shape[1] // 2, chw.shape[2] // 2
    return chw[:, cy - half:cy + half, cx - half:cx + half]


def _run_job(lon, lat, scenes, path):
    for item in _find_scenes(lon, lat, scenes):
        crs, left, top = _patch_geo(item, lon, lat)
        if (_read_window(item, "SCL", left, top, Resampling.nearest) == 0).mean() < 0.2:
            break  # skip swath-edge scenes that are mostly empty
    else:
        raise RuntimeError(f"no usable Sentinel-2 scene within +/-{SEARCH_DAYS} days")

    with ThreadPoolExecutor(len(BANDS)) as pool:
        lr = np.stack(list(pool.map(lambda b: _read_band(item, b, left, top), BANDS)))
    tci = _gpu_products(lr)

    tmp = f"{path}.tmp"  # write then rename: no half-written .tif (paths are unique per job)
    with rasterio.open(tmp, "w", driver="GTiff", width=tci.shape[2],
                       height=tci.shape[1], count=3, dtype="uint8", crs=crs,
                       transform=from_origin(left, top, OUT_RES, OUT_RES), photometric="ycbcr",
                       compress="jpeg", jpeg_quality=92, tiled=True) as dst:
        dst.write(tci)
        dst.update_tags(lon=lon, lat=lat, date=scenes[0].isoformat(), source=item.id)
    os.replace(tmp, path)

    sr_crop = Image.fromarray(_center(tci, round(500 / OUT_RES)).transpose(1, 2, 0))
    lr_crop = Image.fromarray(_to_uint8(torch.from_numpy(_center(lr[RGB], 50))).numpy().transpose(1, 2, 0))
    prev = Image.new("RGB", (2 * sr_crop.width + 8, sr_crop.height), "white")
    prev.paste(lr_crop.resize(sr_crop.size, Image.NEAREST), (0, 0))
    prev.paste(sr_crop, (sr_crop.width + 8, 0))
    return path, prev


# ---------------- Public API ----------------
def run(lonlat, date, out_dir="opensr_out", location=None):
    """Super-resolve every (location, date) pair to a 1 m RGB GeoTIFF.

    lonlat:   (lon, lat), a list of (lon, lat), or CSV text / a CSV file path
              with lon in the first column and lat in the second (header optional).
    date:     "YYYY-MM-DD" or a list of them. With no pass on that exact day,
              the nearest scene within +/- SEARCH_DAYS days is used.
    out_dir:  each call writes to a new {out_dir}/run_<timestamp>/ plus a .zip of it.
    location: name used in the file names: one name for every point, or a list
              with one name per point. Files are output_<date>_<name>.tif, with
              (1), (2)... added when a name repeats for the same date. Without a
              name: output_<date>_run_1.tif, output_<date>_run_2.tif, ...

    Returns one Result per (location, date), sorted by location then date.
    Failed jobs are returned too, with `error` set and `path` None.
    """
    points, dates = _points(lonlat), _dates(date)
    names = _locations(location, len(points))
    _load_model()
    run_dir = os.path.join(out_dir, f"run_{datetime.now():%Y%m%d_%H%M%S}")
    os.makedirs(run_dir, exist_ok=True)

    with ThreadPoolExecutor(len(dates)) as ex:
        scenes = dict(zip(dates, ex.map(lambda d: _search_date(d, points), dates)))

    jobs = [Result(i, lon, lat, d) for (i, (lon, lat)), d in product(enumerate(points, 1), dates)]
    paths = [os.path.join(run_dir, f"{s}.tif") for s in _file_stems(jobs, names)]
    with ThreadPoolExecutor(min(len(jobs), 16)) as ex:
        futs = {ex.submit(_run_job, r.lon, r.lat, scenes[r.date], p): r for r, p in zip(jobs, paths)}
        for n, f in enumerate(as_completed(futs), 1):
            r = futs[f]
            try:
                r.path, r.preview = f.result()
            except Exception as e:
                r.error = str(e)
                print(f"\nFAIL #{r.index} ({r.lon}, {r.lat}) {r.date}: {e}")
            print(f"\r{n}/{len(jobs)} done", end="", flush=True)

    zip_path = f"{run_dir}.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_STORED) as z:  # tifs already compressed
        for name in sorted(os.listdir(run_dir)):
            z.write(os.path.join(run_dir, name), name)
    ok = sum(r.error is None for r in jobs)
    print(f"\rDone: {ok} ok, {len(jobs) - ok} failed. Saved: {run_dir}/  +  {zip_path}")
    _show_previews(jobs)
    return jobs


def _show_previews(jobs):
    """Draw each preview inline when running in a Jupyter/Colab kernel; no-op elsewhere."""
    try:
        from IPython import get_ipython
        from IPython.display import display
    except ImportError:
        return
    if getattr(get_ipython(), "kernel", None) is None:  # plain script or terminal IPython
        return
    for r in jobs:
        if r.preview:
            print(f"#{r.index} ({r.lon}, {r.lat}) {r.date}  {os.path.basename(r.path)}")
            display(r.preview)
