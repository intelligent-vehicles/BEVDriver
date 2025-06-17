import os
import json
import sys
from tqdm import tqdm
from multiprocessing import Pool

# File templates
dt = {
    "topdown": "%04d.jpg",
    "rgb_right": "%04d.jpg",
    "rgb_left": "%04d.jpg",
    "rgb_front": "%04d.jpg",
    "rgb_rear": "%04d.jpg",
    "measurements": "%04d.json",
    "lidar": "%04d.npy",
    "lidar_odd": "%04d.npy",
    "birdview": "%04d.jpg",
    "affordances": "%04d.npy",
    "actors_data": "%04d.json",
    "3d_bbs": "%04d.npy",
    "rgb_full": "%04d.jpg",
    "measurements_full": "%04d.json"
}

def process(task):
    route_dir, end_id, length = task
    for i in range(end_id - length + 6, end_id - 3):
        for key in dt:
            try:
                file_path = os.path.join(route_dir, key, dt[key] % i)
                if os.path.exists(file_path):
                    os.remove(file_path)
            except Exception:
                pass  # Silently ignore errors (e.g., file not found)

if __name__ == "__main__":
    dataset_root = sys.argv[1]
    stat_file = os.path.join(dataset_root, 'blocked_stat.txt')

    tasks = []
    if not os.path.isfile(stat_file):
        print(f"Error: {stat_file} not found")
        sys.exit(1)

    # Read block tasks from file
    with open(stat_file, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 3:
                route_path, end_id, length = parts
                tasks.append([route_path, int(end_id), int(length)])

    # Parallel processing of deletion
    with Pool(8) as p:
        list(tqdm(p.imap(process, tasks), total=len(tasks)))
