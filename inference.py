import os
import os.path as osp
import subprocess
import tyro
from src.config.argument_config import ArgumentConfig
from src.config.inference_config import InferenceConfig
from src.config.crop_config import CropConfig
from src.live_portrait_pipeline import LivePortraitPipeline


def partial_fields(cls, kwargs):
    return cls(**{k: v for k, v in kwargs.items() if hasattr(cls, k)})


def ensure_ffmpeg():
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
        return True
    except Exception:
        ffmpeg_dir = osp.join(os.getcwd(), "ffmpeg")
        if osp.exists(ffmpeg_dir):
            os.environ["PATH"] += os.pathsep + ffmpeg_dir
            try:
                subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
                return True
            except Exception:
                pass
        return False


def validate_paths(*paths):
    for p in paths:
        if not osp.exists(p):
            raise FileNotFoundError(f"Required file not found: {p}")


def main():
    tyro.extras.set_accent_color("bright_cyan")
    args = tyro.cli(ArgumentConfig)

    if not ensure_ffmpeg():
        raise ImportError(
            "FFmpeg is missing. Install it from https://ffmpeg.org/download.html"
        )

    validate_paths(args.source, args.driving)

    inference_cfg = partial_fields(InferenceConfig, args.__dict__)
    crop_cfg = partial_fields(CropConfig, args.__dict__)

    pipeline = LivePortraitPipeline(inference_cfg=inference_cfg, crop_cfg=crop_cfg)
    pipeline.execute(args)


if __name__ == "__main__":
    main()
