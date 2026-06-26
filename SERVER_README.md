# M2T2 grasp server (for the tiptop sim eval)

This repo (vanilla NVlabs/M2T2) was set up as the **grasp-prediction microservice**
that the tiptop planning server calls during perception
(`tiptop/tiptop/perception/m2t2.py`). `server.py` + `pixi.toml` were added here;
the rest is upstream M2T2.

## What was set up

- `pixi.toml` — a self-contained **CUDA 12.8** toolchain (cuda-nvcc, cudart-dev,
  gxx) so `pointnet2_ops` compiles for the RTX 5090 (**sm_120 / Blackwell**).
- `build_server.sh` — installs torch `2.7.1+cu128`, compiles `pointnet2_ops`,
  installs the `m2t2` package + FastAPI, and downloads weights
  (`wentao-yuan/m2t2` → `weights/m2t2.pth`). Already run.
- `server.py` — FastAPI server exposing the tiptop contract:
  - `POST /predict` `{pointcloud:{points,rgb}, num_points, num_runs, mask_thresh, apply_bounds}`
    → `{grasps, grasp_confidence, grasp_contacts}` grouped per object.
  - `GET /health` → `{"status":"healthy"}` once the model is loaded.

It reproduces M2T2's `demo.py` pick-task inference (centered-xyz + ImageNet-RGB
inputs, sample to `num_points`, run `num_runs` times, accumulate per-object grasps).

## Run it manually

```bash
cd M2T2
pixi run python server.py --port 8123      # default port tiptop expects
```

## Rebuild the env (if needed)

```bash
cd M2T2 && bash build_server.sh
```

## How full_eval uses it

`droid-sim-evals/full_eval.py` launches this server automatically before the
tiptop phase, waits for `/health`, runs the tiptop rollouts, then kills it
(`--launch-perception`, on by default; `--m2t2-port 8123`). Nothing extra to do —
just run `full_eval.py` with `GEMINI_API_KEY` exported (tiptop perception also
needs Gemini).
