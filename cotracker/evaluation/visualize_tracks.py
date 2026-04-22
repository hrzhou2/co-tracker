import argparse
import os

import numpy as np
import torch
from PIL import Image

from cotracker.utils.visualizer import Visualizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", type=str, required=True, help="image dir")
    parser.add_argument("--track_dir", type=str, required=True, help="tracking result dir")
    parser.add_argument("--out_dir", type=str, default=None, help="out dir (default: track_dir)")
    parser.add_argument("--query_frame", type=int, default=0, help="query frame index")
    args = parser.parse_args()

    frame_names = sorted([
        p for p in os.listdir(args.image_dir)
        if os.path.splitext(p)[-1] in [".jpg", ".png"]
    ])
    num_frames = len(frame_names)

    # load video frames
    video = np.stack([
        np.array(Image.open(os.path.join(args.image_dir, f))) for f in frame_names
    ])
    video = torch.from_numpy(video).permute(0, 3, 1, 2).float()  # (T, 3, H, W)

    # load predicted tracks for the query frame
    start_frame = os.path.splitext(frame_names[args.query_frame])[0]
    outputs = np.stack([
        np.load(os.path.join(args.track_dir, f"{start_frame}_{os.path.splitext(frame_names[j])[0]}.npy"))
        for j in range(num_frames)
    ], axis=1)  # (N, T, 4): x, y, vis, conf

    # (N, T, 4) -> tracks (1, T, N, 2), visibility (1, T, N)
    pred_tracks = torch.from_numpy(outputs[:, :, :2]).permute(1, 0, 2).unsqueeze(0)  # (1, T, N, 2)
    pred_visibility = torch.from_numpy(outputs[:, :, 2] * outputs[:, :, 3] > 0.5).permute(1, 0).unsqueeze(0)  # (1, T, N), bool

    out_dir = args.out_dir if args.out_dir is not None else os.path.join(args.track_dir, "vis")
    os.makedirs(out_dir, exist_ok=True)
    vis = Visualizer(save_dir=out_dir, show_first_frame=0, linewidth=1)
    vis.visualize(
        video[None],
        pred_tracks,
        pred_visibility,
        filename=start_frame,
        query_frame=args.query_frame,
        save_frames=True,
    )


# Example:
# python ./cotracker/evaluation/visualize_tracks.py --image_dir /path/to/images/ --track_dir ./outputs/horsejump-high --out_dir ./outputs/test --query_frame 0
if __name__ == "__main__":
    main()