import argparse
import os

import numpy as np
import torch
from tqdm import tqdm

from cotracker.evaluation.compute_tracks import build_predictor, predict_one_frame, read_video


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", type=str, required=True, help="image dir")
    parser.add_argument("--mask_dir", type=str, default=None, help="mask dir (if not provided, predict all grid points)")
    parser.add_argument("--out_dir", type=str, required=True, help="out dir")
    parser.add_argument("--grid_size", type=int, default=4, help="grid size")
    parser.add_argument("--chunk_size", type=int, default=384, help="number of points per inference chunk")
    parser.add_argument("--model_type", type=str, default="offline", choices=["online", "offline"])
    parser.add_argument("--ckpt_path", type=str, default=None, help="override checkpoint path")
    parser.add_argument("--shuffle", action="store_true", help="enable shuffling of points before chunking")
    args = parser.parse_args()

    frame_names = sorted([x for x in os.listdir(args.image_dir) if x.endswith((".png", ".jpg"))])
    num_frames = len(frame_names)
    os.makedirs(args.out_dir, exist_ok=True)

    # # Check if everything is already done
    # done = all(
    #     os.path.exists(
    #         f"{args.out_dir}/{os.path.splitext(frame_names[t])[0]}_{os.path.splitext(frame_names[j])[0]}.npy"
    #     )
    #     for t in range(num_frames)
    #     for j in range(num_frames)
    # )
    # if done:
    #     print("Already done")
    #     return

    assert args.model_type == "offline", "Online model is broken: it cannot track backward before the query frame."
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
        print(f"{video.shape=} {masks.shape=}")
    else:
        masks = None
        print(f"{video.shape=} masks=None")

    frames = torch.from_numpy(video).to(device)
    frames = frames.permute(0, 3, 1, 2)[None].float()  # (1, T, 3, H, W)

    for t in tqdm(range(num_frames), desc="query frames"):
        name_t = os.path.splitext(frame_names[t])[0]
        # if all(
        #     os.path.exists(f"{args.out_dir}/{name_t}_{os.path.splitext(frame_names[j])[0]}.npy")
        #     for j in range(num_frames)
        # ):
        #     print(f"Already computed tracks for query_frame={t}")
        #     continue

        predict_one_frame(
            predictor, frames, masks, frame_names,
            t=t,
            out_dir=args.out_dir,
            grid_size=args.grid_size,
            model_type=args.model_type,
            device=device,
            chunk_size=args.chunk_size,
            shuffle_chunks=args.shuffle,
        )


# Example:
# python ./cotracker/evaluation/compute_tracks_all.py --image_dir /path/to/images/ --mask_dir /path/to/masks/ --out_dir ./outputs/test --model_type offline
if __name__ == "__main__":
    main()
