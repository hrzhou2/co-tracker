import argparse
import glob
import os

import imageio
import mediapy as media
import numpy as np
import torch
from tqdm import tqdm
from dataclasses import dataclass, field
from omegaconf import OmegaConf

from cotracker.models.evaluation_predictor import EvaluationPredictor
from cotracker.models.build_cotracker import build_cotracker
from cotracker.predictor import CoTrackerOnlinePredictor
from cotracker.utils.visualizer import Visualizer


@dataclass(eq=False)
class DefaultConfig:
    # Directory where all outputs of the experiment will be saved.
    exp_dir: str = "./outputs"

    # Name of the dataset to be used for the evaluation.
    dataset_name: str = "tapvid_davis_first"
    # The root directory of the dataset.
    dataset_root: str = "./"
    # Path to the pre-trained model checkpoint to be used for the evaluation.
    # The default value is the path to a specific CoTracker model checkpoint.
    checkpoint: str = "./checkpoints/scaled_online.pth"
    # EvaluationPredictor parameters
    # The size (N) of the support grid used in the predictor.
    # The total number of points is (N*N).
    grid_size: int = 5
    # The size (N) of the local support grid.
    local_grid_size: int = 8
    num_uniformly_sampled_pts: int = 0
    sift_size: int = 0
    # A flag indicating whether to evaluate one ground truth point at a time.
    single_point: bool = False
    offline_model: bool = False
    window_len: int = 16
    # The number of iterative updates for each sliding window.
    n_iters: int = 6

    seed: int = 0
    gpu_idx: int = 0
    local_extent: int = 50

    v2: bool = False

    # Override hydra's working directory to current working dir,
    # also disable storing the .hydra logs:
    hydra: dict = field(
        default_factory=lambda: {
            "run": {"dir": "."},
            "output_subdir": None,
        }
    )


def read_video(folder_path):
    frame_paths = sorted(glob.glob(os.path.join(folder_path, "*")))
    video = np.stack([imageio.imread(frame_path) for frame_path in frame_paths])
    print(f"{video.shape=} {video.dtype=} {video.min()=} {video.max()=}")
    video = media._VideoArray(video)
    return video


def load_config(yaml_path: str) -> DefaultConfig:
    yaml_cfg = OmegaConf.load(yaml_path)
    if "defaults" in yaml_cfg:
        del yaml_cfg["defaults"]
    default_cfg = OmegaConf.structured(DefaultConfig)
    merged_cfg = OmegaConf.merge(default_cfg, yaml_cfg)
    return OmegaConf.to_object(merged_cfg)


def build_predictor(model_type: str, ckpt_path: str = None):
    ## Load model
    if model_type == "online":
        cfg = load_config("./cotracker/evaluation/configs/pred_davis_online.yaml")
    elif model_type == "offline":
        cfg = load_config("./cotracker/evaluation/configs/pred_davis_offline.yaml")
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    if ckpt_path is not None:
        cfg.checkpoint = ckpt_path

    cotracker_model = build_cotracker(
        cfg.checkpoint, offline=cfg.offline_model, window_len=cfg.window_len, v2=cfg.v2
    )

    # Creating the EvaluationPredictor object
    predictor = EvaluationPredictor(
        cotracker_model,
        grid_size=cfg.grid_size,
        local_grid_size=cfg.local_grid_size,
        sift_size=cfg.sift_size,
        single_point=cfg.single_point,
        num_uniformly_sampled_pts=cfg.num_uniformly_sampled_pts,
        n_iters=cfg.n_iters,
        local_extent=cfg.local_extent,
        interp_shape=(384, 512),
    )

    if torch.cuda.is_available():
        predictor.model = predictor.model.cuda()

    return predictor, cfg


def predict_one_frame(
    predictor, frames, masks, frame_names, t, out_dir,
    grid_size, model_type, device, chunk_size=384, shuffle_chunks=False, vis_dir=None,
):
    """Compute and save tracks for all query points in frame t across all frames.

    Saves {out_dir}/{name_t}_{name_j}.npy for each target frame j.
    Each file has shape (N, 4): x, y, visibility, confidence.

    Notes on chunk_size and shuffle_chunks:
    - CoTracker processes all queries in a chunk together via transformer attention, so points
      within a chunk influence each other. Chunk_size can affect the output tracks.
    - Without shuffle (default), points are chunked row by row (from np.mgrid), so each chunk
      is a horizontal strip — spatially coherent, producing smoother tracks for nearby points.
    """
    num_frames = frames.shape[1]
    height, width = frames.shape[3], frames.shape[4]

    y, x = np.mgrid[0:height:grid_size, 0:width:grid_size]

    all_points = np.stack([t * np.ones_like(y), x, y], axis=-1)
    if masks is None:
        in_mask = np.ones(y.shape, dtype=bool)
    else:
        in_mask = masks[t][y, x] > 0.5
    all_points_t = all_points[in_mask]
    print(f"{all_points.shape=} {all_points_t.shape=} {t=}")

    if shuffle_chunks:
        perm = np.random.permutation(len(all_points_t))
        all_points_t = all_points_t[perm]

    outputs = []
    if len(all_points_t) > 0:
        chunks = [all_points_t[i:i + chunk_size] for i in range(0, len(all_points_t), chunk_size)]
        for chunk in tqdm(chunks, leave=False, desc="points"):
            n_valid = len(chunk)
            if n_valid < chunk_size and len(all_points_t) - n_valid > 0:
                # pad last chunk with random points from outside it so all chunks are exactly chunk_size
                pad_pts = all_points_t[np.random.choice(len(all_points_t) - n_valid, chunk_size - n_valid, replace=False)]
                chunk = np.concatenate([chunk, pad_pts], axis=0)
            points = torch.from_numpy(chunk.astype(np.float32))[None].to(device)

            # predict
            with torch.inference_mode():
                if model_type == "online":
                    # Original evaluate.py uses CoTrackerOnlinePredictor for online mode
                    online_model = CoTrackerOnlinePredictor(checkpoint=None)
                    online_model.model = predictor.model
                    online_model.step = predictor.model.window_len // 2
                    online_model(
                        video_chunk=frames,  # (B, T, 3, H, W), float32, 0-255
                        is_first_step=True,
                        queries=points,  # (B, N, 3), float32, xy
                        add_support_grid=False,
                    )
                    # Process the video
                    for idx in range(0, frames.shape[1] - online_model.step, online_model.step):
                        pred_tracks = online_model(
                            video_chunk=frames[:, idx: idx + online_model.step * 2],
                            add_support_grid=False,
                            grid_size=0,
                            return_conf=True,
                        )  # (B, T, N, 2), (B, T, N)
                elif model_type == "offline":
                    # Original evaluate.py uses EvaluationPredictor for offline mode
                    pred_tracks = predictor(frames, points, return_conf=True)
                else:
                    raise ValueError(f"Unknown model type: {model_type}")

            # save predicted results
            tracks, visibilities, confidences = (
                pred_tracks[0].squeeze(0).permute(1, 0, 2).detach().cpu().numpy(),  # (N, T, 2), input resolution
                pred_tracks[1].squeeze(0).permute(1, 0).detach().cpu().numpy(),  # (N, T)
                pred_tracks[2].squeeze(0).permute(1, 0).detach().cpu().numpy(),  # (N, T)
            )
            outputs.append(
                np.concatenate([tracks, visibilities[..., None], confidences[..., None]], axis=-1)[:n_valid]
            )
        outputs = np.concatenate(outputs, axis=0)  # (N, T, 4)
        if shuffle_chunks:
            unshuffled = np.empty_like(outputs)
            unshuffled[perm] = outputs
            outputs = unshuffled
    else:
        outputs = np.zeros((0, num_frames, 4), dtype=np.float32)

    name_t = os.path.splitext(frame_names[t])[0]
    for j in range(num_frames):
        if j == t:
            original_query_points = np.stack([x[in_mask], y[in_mask]], axis=-1)
            outputs[:, j, :2] = original_query_points
        name_j = os.path.splitext(frame_names[j])[0]
        np.save(f"{out_dir}/{name_t}_{name_j}.npy", outputs[:, j])

    if vis_dir is not None and len(outputs) > 0:
        os.makedirs(vis_dir, exist_ok=True)
        # outputs: (N, T, 4) -> tracks (1, T, N, 2), visibility (1, T, N)
        pred_tracks = torch.from_numpy(outputs[:, :, :2]).permute(1, 0, 2).unsqueeze(0)  # (1, T, N, 2)
        pred_visibility = torch.from_numpy(outputs[:, :, 2] * outputs[:, :, 3] > 0.5).permute(1, 0).unsqueeze(0)  # (1, T, N), bool
        vis = Visualizer(save_dir=vis_dir, show_first_frame=0)
        vis.visualize(
            frames.cpu(),
            pred_tracks,
            pred_visibility,
            segm_mask=torch.from_numpy(masks).float().unsqueeze(0) if masks is not None else None,  # (1, T, H, W)
            filename=name_t,
            query_frame=t,
            save_frames=True,
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", type=str, required=True, help="image dir")
    parser.add_argument("--mask_dir", type=str, default=None, help="mask dir (if not provided, predict all grid points)")
    parser.add_argument("--out_dir", type=str, required=True, help="out dir")
    parser.add_argument("--query_frame", type=int, required=True, help="query frame index")
    parser.add_argument("--grid_size", type=int, default=4, help="grid size")
    parser.add_argument("--chunk_size", type=int, default=384, help="number of points per inference chunk")
    parser.add_argument("--model_type", type=str, default="offline", choices=["online", "offline"], help="model type")
    parser.add_argument("--ckpt_path", type=str, default=None, help="override checkpoint path")
    parser.add_argument("--vis", action="store_true", help="if set, save track visualizations to {out_dir}/vis")
    parser.add_argument("--shuffle", action="store_true", help="enable shuffling of points before chunking")
    args = parser.parse_args()

    frame_names = sorted([x for x in os.listdir(args.image_dir) if x.endswith((".png", ".jpg"))])
    os.makedirs(args.out_dir, exist_ok=True)

    # # Skip if already done
    # name_t = os.path.splitext(frame_names[args.query_frame])[0]
    # file_matches = glob.glob(f"{args.out_dir}/{name_t}_*.npy")
    # if len(file_matches) == len(frame_names):
    #     print(f"Already computed tracks for query_frame={args.query_frame}")
    #     return

    predictor, cfg = build_predictor(args.model_type, ckpt_path=args.ckpt_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Setting the random seeds
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    video = read_video(args.image_dir)
    num_frames, height, width = video.shape[0:3]
    if args.mask_dir is not None:
        masks = read_video(args.mask_dir)
        masks = (masks.reshape((num_frames, height, width, -1)) > 0).any(axis=-1)
    else:
        masks = None
    print(f"{video.shape=}")

    frames = torch.from_numpy(video).to(device)
    frames = frames.permute(0, 3, 1, 2)[None].float()  # (1, T, 3, H, W)

    predict_one_frame(
        predictor, frames, masks, frame_names,
        t=args.query_frame,
        out_dir=args.out_dir,
        grid_size=args.grid_size,
        model_type=args.model_type,
        device=device,
        chunk_size=args.chunk_size,
        shuffle_chunks=args.shuffle,
        vis_dir=os.path.join(args.out_dir, "vis") if args.vis else None,
    )


# Example: python ./cotracker/evaluation/compute_tracks.py --image_dir /path/to/images/ --mask_dir /path/to/masks/ --out_dir ./outputs/test --query_frame 0 --model_type offline --vis
if __name__ == "__main__":
    main()
