"""M2T2 grasp-prediction HTTP server.

Exposes the API the tiptop perception pipeline expects (see
tiptop/tiptop/perception/m2t2.py):

  POST /predict
    body: {"pointcloud": {"points": [[x,y,z],...], "rgb": [[r,g,b],...]},
           "num_points": int, "num_runs": int, "mask_thresh": float,
           "apply_bounds": bool}
    -> {"grasps": [[<4x4>,...] per object],
        "grasp_confidence": [[float,...] per object],
        "grasp_contacts": [[[x,y,z],...] per object]}

  GET /health -> {"status": "healthy"}  (only once the model is loaded)

Reproduces the inference in M2T2's demo.py (pick task, world_coord=True):
build [centered_xyz, imagenet_rgb] inputs, sample to num_points, run the model
num_runs times, and accumulate per-object grasp outputs.

Run inside the M2T2 pixi env:
    pixi run python server.py --port 8123
"""

import argparse
import logging

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI
from omegaconf import OmegaConf
from pydantic import BaseModel

from m2t2.dataset import collate
from m2t2.dataset_utils import sample_points
from m2t2.m2t2 import M2T2
from m2t2.train_utils import to_cpu, to_gpu

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("m2t2_server")

# ImageNet RGB normalization (matches m2t2.dataset_utils.normalize_rgb).
_RGB_MEAN = torch.tensor([0.485, 0.456, 0.406])
_RGB_STD = torch.tensor([0.229, 0.224, 0.225])

_MODEL: M2T2 | None = None
_CFG = None
_SURFACE_RANGE = 0.02  # table-flatten band (config.yaml eval.surface_range)


class PointCloud(BaseModel):
    points: list
    rgb: list


class PredictRequest(BaseModel):
    pointcloud: PointCloud
    num_points: int = 16384
    num_runs: int = 5
    mask_thresh: float = 0.035
    apply_bounds: bool = True


def load_model(config_path: str, checkpoint_path: str) -> None:
    global _MODEL, _CFG
    cfg = OmegaConf.load(config_path)
    cfg.eval.checkpoint = checkpoint_path
    model = M2T2.from_config(cfg.m2t2)
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    _MODEL = model.cuda().eval()
    _CFG = cfg
    logger.info(f"Loaded M2T2 from {checkpoint_path} (num_points={cfg.data.num_points})")


def _build_base_data(xyz: torch.Tensor, rgb01: torch.Tensor, apply_bounds: bool) -> dict:
    """Build the pick-task data dict (pre-sampling), mirroring load_rgb_xyz."""
    rgb_norm = (rgb01 - _RGB_MEAN) / _RGB_STD
    if apply_bounds:
        # Flatten points near the table plane to z=0 (load_rgb_xyz surface_range).
        xyz = xyz.clone()
        xyz[xyz[:, 2].abs() < _SURFACE_RANGE, 2] = 0.0
    return {
        "inputs": torch.cat([xyz - xyz.mean(dim=0), rgb_norm], dim=1),  # (N,6)
        "points": xyz,  # (N,3) world frame, uncentered
        "seg": torch.zeros(xyz.shape[0], dtype=torch.long),  # dummy; unused by infer
        "cam_pose": torch.eye(4),
        "object_inputs": torch.rand(1024, 6),  # pick-mode placeholder (zeroed by task_is_place)
        "ee_pose": torch.eye(4),
        "bottom_center": torch.zeros(3),
        "object_center": torch.zeros(3),
        "task": "pick",
    }


@torch.no_grad()
def predict_grasps(xyz: np.ndarray, rgb01: np.ndarray, num_points: int, num_runs: int,
                   mask_thresh: float, apply_bounds: bool) -> dict:
    assert _MODEL is not None and _CFG is not None, "model not loaded"
    eval_cfg = _CFG.eval.copy()
    eval_cfg.mask_thresh = float(mask_thresh)

    base = _build_base_data(
        torch.from_numpy(np.ascontiguousarray(xyz)).float(),
        torch.from_numpy(np.ascontiguousarray(rgb01)).float(),
        apply_bounds,
    )
    inputs_full, points_full, seg_full = base["inputs"], base["points"], base["seg"]

    acc = {"grasps": [], "grasp_confidence": [], "grasp_contacts": []}
    for _ in range(max(1, int(num_runs))):
        pt_idx = sample_points(points_full, int(num_points))
        data = dict(base)
        data["inputs"] = inputs_full[pt_idx]
        data["points"] = points_full[pt_idx]
        data["seg"] = seg_full[pt_idx]
        data_batch = collate([data])
        to_gpu(data_batch)
        out = _MODEL.infer(data_batch, eval_cfg)
        to_cpu(out)
        for key in acc:
            # out[key] is per-batch; [0] is the per-object list for our single sample.
            acc[key].extend(out[key][0])

    return {
        "grasps": [np.asarray(g).reshape(-1, 4, 4).tolist() for g in acc["grasps"]],
        "grasp_confidence": [np.asarray(c).reshape(-1).tolist() for c in acc["grasp_confidence"]],
        "grasp_contacts": [np.asarray(c).reshape(-1, 3).tolist() for c in acc["grasp_contacts"]],
    }


app = FastAPI()


@app.get("/health")
def health():
    return {"status": "healthy" if _MODEL is not None else "loading"}


@app.post("/predict")
def predict(req: PredictRequest):
    xyz = np.asarray(req.pointcloud.points, dtype=np.float32).reshape(-1, 3)
    rgb = np.asarray(req.pointcloud.rgb, dtype=np.float32).reshape(-1, 3)
    if xyz.shape[0] == 0:
        return {"grasps": [], "grasp_confidence": [], "grasp_contacts": []}
    result = predict_grasps(xyz, rgb, req.num_points, req.num_runs, req.mask_thresh, req.apply_bounds)
    n_obj = len(result["grasps"])
    n_grasps = sum(len(g) for g in result["grasps"])
    logger.info(f"/predict: {xyz.shape[0]} pts -> {n_obj} objects, {n_grasps} grasps")
    return result


def main():
    ap = argparse.ArgumentParser(description="M2T2 grasp server")
    ap.add_argument("--port", type=int, default=8123)
    ap.add_argument("--host", type=str, default="0.0.0.0")
    ap.add_argument("--config", type=str, default="config.yaml")
    ap.add_argument("--checkpoint", type=str, default="weights/m2t2.pth")
    args = ap.parse_args()
    load_model(args.config, args.checkpoint)
    logger.info(f"Serving M2T2 on {args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
