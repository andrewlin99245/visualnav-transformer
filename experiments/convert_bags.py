"""
Convert ROS bag files to ViNT dataset format.

Target format (same as go_stanford/scand):
  dataset_name/
    traj_folder_1/
      0.jpg
      1.jpg
      ...
      traj_data.pkl   # {"position": np.array([N, 2]), "yaw": np.array([N, 1])}
    traj_folder_2/
    ...

For each bag:
  - Sync images to odometry timestamps (nearest neighbor)
  - Save numbered images as .jpg
  - Save position (x, y) and yaw to traj_data.pkl

Usage:
  cd experiments && python convert_bags.py
  cd experiments && python convert_bags.py --bag-dir /path/to/bags --output-dir /path/to/output
"""

import argparse
import math
import os
import pickle
import sys
from pathlib import Path

import cv2
import numpy as np
import rosbag

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def quaternion_to_yaw(ori):
    siny = 2.0 * (ori.w * ori.z + ori.x * ori.y)
    cosy = 1.0 - 2.0 * (ori.y * ori.y + ori.z * ori.z)
    return math.atan2(siny, cosy)


def convert_bag(bag_path, output_dir, image_topic="/usb_cam/image_raw/compressed",
                odom_topic="/laser_odometry", min_distance=0.12):
    """Convert a single bag file to ViNT dataset format.

    Subsamples by distance: only keeps a frame when the robot has moved
    at least min_distance (meters) since the last kept frame.
    """
    bag = rosbag.Bag(str(bag_path))

    # Read all odometry messages
    odom_times = []
    positions = []
    yaws = []
    for topic, msg, t in bag.read_messages(topics=[odom_topic]):
        odom_times.append(t.to_sec())
        pos = msg.pose.pose.position
        positions.append([pos.x, pos.y])
        yaws.append(quaternion_to_yaw(msg.pose.pose.orientation))

    if not odom_times:
        print(f"  WARNING: No odometry in {bag_path.name}, skipping")
        bag.close()
        return 0

    odom_times = np.array(odom_times)
    positions = np.array(positions)
    yaws = np.array(yaws)

    # Read all images, sync to nearest odometry timestamp
    # Subsample by distance: only keep frame if robot moved >= min_distance
    output_dir.mkdir(parents=True, exist_ok=True)
    img_count = 0
    synced_positions = []
    synced_yaws = []
    last_odom_idx = -1
    last_kept_pos = None

    for topic, msg, t in bag.read_messages(topics=[image_topic]):
        img_time = t.to_sec()
        # Find nearest odometry timestamp
        odom_idx = np.argmin(np.abs(odom_times - img_time))
        time_diff = abs(odom_times[odom_idx] - img_time)

        # Skip if time difference too large (>0.2s) or same odom as last image
        if time_diff > 0.2:
            continue
        if odom_idx == last_odom_idx:
            continue
        last_odom_idx = odom_idx

        # Distance-based subsampling
        cur_pos = positions[odom_idx]
        if last_kept_pos is not None:
            dist = np.linalg.norm(cur_pos - last_kept_pos)
            if dist < min_distance:
                continue

        # Decode and save image
        img = cv2.imdecode(np.frombuffer(msg.data, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            continue
        cv2.imwrite(str(output_dir / f"{img_count}.jpg"), img)
        synced_positions.append(cur_pos)
        synced_yaws.append(yaws[odom_idx])
        last_kept_pos = cur_pos
        img_count += 1

    bag.close()

    if img_count < 2:
        print(f"  WARNING: Only {img_count} synced frames in {bag_path.name}, skipping")
        return 0

    # Save traj_data.pkl
    synced_positions = np.array(synced_positions)  # [N, 2]
    synced_yaws = np.array(synced_yaws)  # [N]
    # Match go_stanford format: position is [N, 2] object array, yaw is [N] object array
    # But actually, let's check what the dataset loader expects
    traj_data = {
        "position": synced_positions,
        "yaw": synced_yaws.reshape(-1, 1),
    }
    with open(output_dir / "traj_data.pkl", "wb") as f:
        pickle.dump(traj_data, f)

    return img_count


def main():
    parser = argparse.ArgumentParser(description="Convert ROS bags to ViNT dataset format")
    parser.add_argument("--bag-dir", type=str,
                        default=str(PROJECT_ROOT / "datasets" / "mist_bags"))
    parser.add_argument("--output-dir", type=str,
                        default=str(PROJECT_ROOT / "datasets" / "mist_bags_converted"))
    parser.add_argument("--image-topic", type=str, default="/usb_cam/image_raw/compressed")
    parser.add_argument("--odom-topic", type=str, default="/laser_odometry")
    args = parser.parse_args()

    bag_dir = Path(args.bag_dir)
    output_dir = Path(args.output_dir)
    bag_files = sorted(bag_dir.glob("*.bag"))

    print(f"Found {len(bag_files)} bag files in {bag_dir}")
    print(f"Output directory: {output_dir}")

    total_frames = 0
    total_trajs = 0
    for bag_path in bag_files:
        traj_name = bag_path.stem
        traj_dir = output_dir / traj_name
        print(f"\n  Converting {bag_path.name}...")
        n = convert_bag(bag_path, traj_dir, args.image_topic, args.odom_topic)
        if n > 0:
            total_trajs += 1
            total_frames += n
            path_len = np.linalg.norm(
                np.diff(np.load(traj_dir / "traj_data.pkl", allow_pickle=True) if False else
                        pickle.load(open(traj_dir / "traj_data.pkl", "rb"))["position"], axis=0),
                axis=1).sum()
            print(f"    {n} frames, path length: {path_len:.2f}m")

    print(f"\n=== Done ===")
    print(f"  {total_trajs} trajectories, {total_frames} total frames")
    print(f"  Output: {output_dir}")


if __name__ == "__main__":
    main()
