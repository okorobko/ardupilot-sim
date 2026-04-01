#!/usr/bin/env python3
"""
YOLOv8n two-phase training entry point for military vehicle detection.

Phase 1: Frozen backbone warmup (10 epochs, LR 0.001, freeze first 10 layers)
Phase 2: Full fine-tune (150+ epochs, LR 0.01, cosine decay)
"""

import argparse
import sys
from pathlib import Path

import yaml
from ultralytics import YOLO


def load_config(config_path: Path) -> dict:
    """Load training configuration from YAML file."""
    if not config_path.exists():
        print(f"[WARN] Config file not found: {config_path}, using defaults")
        return {}
    with open(config_path, "r") as f:
        return yaml.safe_load(f) or {}


def phase_warmup(
    model: YOLO,
    dataset_yaml: str,
    project: str,
    name: str,
    cfg: dict,
) -> Path:
    """Phase 1: frozen backbone warmup."""
    print("\n" + "=" * 60)
    print("PHASE 1 — Frozen Backbone Warmup")
    print("=" * 60)

    warmup_cfg = cfg.get("warmup", {})
    epochs = warmup_cfg.get("epochs", 10)
    lr0 = warmup_cfg.get("lr0", 0.001)
    freeze = warmup_cfg.get("freeze", 10)

    results = model.train(
        data=dataset_yaml,
        epochs=epochs,
        lr0=lr0,
        freeze=freeze,
        imgsz=cfg.get("imgsz", 640),
        batch=cfg.get("batch", 16),
        workers=cfg.get("workers", 8),
        device=cfg.get("device", None),
        project=project,
        name=f"{name}_warmup",
        exist_ok=True,
        # Aerial-optimised augmentations
        degrees=cfg.get("degrees", 180),
        flipud=cfg.get("flipud", 0.5),
        mosaic=cfg.get("mosaic", 1.0),
        mixup=cfg.get("mixup", 0.1),
        verbose=True,
    )

    best_pt = Path(project) / f"{name}_warmup" / "weights" / "best.pt"
    if not best_pt.exists():
        print(f"[ERROR] Warmup best weights not found at {best_pt}")
        sys.exit(1)

    print(f"Phase 1 complete. Best weights: {best_pt}")
    return best_pt


def phase_full(
    model: YOLO,
    dataset_yaml: str,
    project: str,
    name: str,
    cfg: dict,
) -> Path:
    """Phase 2: full fine-tune with cosine decay."""
    print("\n" + "=" * 60)
    print("PHASE 2 — Full Fine-Tune")
    print("=" * 60)

    full_cfg = cfg.get("finetune", {})
    epochs = full_cfg.get("epochs", 150)
    lr0 = full_cfg.get("lr0", 0.01)
    lrf = full_cfg.get("lrf", 0.01)  # cosine decay final LR ratio

    results = model.train(
        data=dataset_yaml,
        epochs=epochs,
        lr0=lr0,
        lrf=lrf,
        imgsz=cfg.get("imgsz", 640),
        batch=cfg.get("batch", 16),
        workers=cfg.get("workers", 8),
        device=cfg.get("device", None),
        project=project,
        name=f"{name}_full",
        exist_ok=True,
        # Aerial-optimised augmentations
        degrees=cfg.get("degrees", 180),
        flipud=cfg.get("flipud", 0.5),
        mosaic=cfg.get("mosaic", 1.0),
        mixup=cfg.get("mixup", 0.1),
        cos_lr=True,
        verbose=True,
    )

    best_pt = Path(project) / f"{name}_full" / "weights" / "best.pt"
    if not best_pt.exists():
        print(f"[ERROR] Fine-tune best weights not found at {best_pt}")
        sys.exit(1)

    print(f"Phase 2 complete. Best weights: {best_pt}")
    return best_pt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="YOLOv8n two-phase training for military vehicle detection"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "configs" / "train_config.yaml",
        help="Path to training config YAML (default: ../configs/train_config.yaml)",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "configs" / "dataset.yaml",
        help="Path to dataset YAML (default: ../configs/dataset.yaml)",
    )
    parser.add_argument(
        "--phase",
        choices=["warmup", "full", "both"],
        default="both",
        help="Training phase to run (default: both)",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Path to checkpoint to resume from",
    )
    parser.add_argument(
        "--project",
        type=str,
        default=str(Path(__file__).resolve().parent.parent / "models"),
        help="Output directory (default: ../models)",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="mil_vehicle_v1",
        help="Run name (default: mil_vehicle_v1)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    dataset_yaml = str(args.dataset.resolve())
    if not args.dataset.exists():
        print(f"[ERROR] Dataset config not found: {dataset_yaml}")
        sys.exit(1)

    print(f"Config:  {args.config}")
    print(f"Dataset: {dataset_yaml}")
    print(f"Phase:   {args.phase}")
    print(f"Project: {args.project}")
    print(f"Name:    {args.name}")

    # ---- Phase 1: Warmup ----
    if args.phase in ("warmup", "both"):
        if args.resume and args.phase == "warmup":
            model = YOLO(str(args.resume))
        else:
            base_weights = cfg.get("base_weights", "yolov8n.pt")
            model = YOLO(base_weights)

        warmup_best = phase_warmup(model, dataset_yaml, args.project, args.name, cfg)
    else:
        warmup_best = None

    # ---- Phase 2: Full fine-tune ----
    if args.phase in ("full", "both"):
        if args.resume and args.phase == "full":
            checkpoint = str(args.resume)
        elif warmup_best is not None:
            checkpoint = str(warmup_best)
        else:
            print("[ERROR] --resume is required when running --phase=full standalone")
            sys.exit(1)

        model = YOLO(checkpoint)
        final_best = phase_full(model, dataset_yaml, args.project, args.name, cfg)
        print(f"\nTraining complete. Final weights: {final_best}")


if __name__ == "__main__":
    main()
