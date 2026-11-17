# opensr

Sentinel-2 L2A (Planetary Computer) 10 m to 1 m RGB super-resolution for any
set of locations and dates. The SEN2-SR model takes 10 m to 2.5 m; bicubic
resampling plus sharpening then puts it on a 1 m grid. Detail finer than
~2.5 m is not recovered, so treat 1 m as a pixel size, not true resolution.

Each (location, date) pair becomes a 5.12 km GeoTIFF (5120 x 5120 px at 1 m);
a run folder and a zip of it are written per call.

## Install

```bash
pip install git+https://github.com/Eavheang/openSR.git
# or, from a local checkout
pip install -e .
```

## Use

```python
import opensr.inferutils

# one location, one date
opensr.inferutils.run((105.392986, 14.121299), "2026-11-08")

# many locations x many dates (every combination is processed)
results = opensr.inferutils.run(
    [(105.392986, 14.121299), (104.9160, 11.5564)],
    ["2026-10-14", "2026-10-29", "2026-11-08"],
    out_dir="/content/s2dr3_out",
)

# CSV text or a CSV file path also work (lon first, lat second, header optional)
opensr.inferutils.run("points.csv", ["2026-11-08"])
```

`run` returns one `Result` per (location, date) with `path` (GeoTIFF),
`preview` (PIL image: 10 m input | 1 m output, centre 1 km) and `error`
(set when that job failed). In Jupyter or Colab, `run` also draws every
preview inline under the progress output; in a plain script it only returns
the results.

The model weights download once to `~/.cache/opensr/` on the first call.
A GPU is used when available; CPU works but is slow.
