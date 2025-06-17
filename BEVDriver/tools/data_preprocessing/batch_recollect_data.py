import os
import json
import sys
from tqdm import tqdm
from multiprocessing import Pool

verbose = False

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

def process(relative_route):
    full_route = os.path.join(dataset_root, relative_route)

    try:
        frames = len(os.listdir(os.path.join(full_route, "measurements")))
    except FileNotFoundError:
        print(f"[!] Missing 'measurements' folder: {full_route}")
        return

    for folder in dt:
        file_pattern = dt[folder]
        folder_path = os.path.join(full_route, folder)

        try:
            files = os.listdir(folder_path)
        except FileNotFoundError:
            continue

        # Track and sort valid frame IDs
        fs = []
        for file in files:
            try:
                fs.append(int(file.split('.')[0]))
            except ValueError:
                continue

        fs.sort()

        if len(fs) != frames:
            print(f"[!] Mismatch in {folder_path}: expected {frames}, found {len(fs)}")

        # Rename frames to be continuous (e.g., 0 → N-1)
        for i in range(len(fs)):
            if i == fs[i]:
                continue  # already correct
            try:
                src = os.path.join(folder_path, file_pattern % fs[i])
                dst = os.path.join(folder_path, file_pattern % i)
                if verbose:
                    print(f"Renaming {src} → {dst}")
                os.rename(src, dst)
            except Exception as e:
                print(f"[X] Rename error: {src} → {dst}: {e}")

if __name__ == "__main__":
    dataset_root = sys.argv[1]
    blocked_file = os.path.join(dataset_root, 'blocked_stat.txt')

    routes = []
    with open(blocked_file, 'r') as f:
        for line in f:
            route_path = line.strip().split()[0]  # Only take the path
            routes.append(route_path)

    # Remove duplicates
    routes = list(set(routes))

    with Pool(4) as p:
        list(tqdm(p.imap(process, routes), total=len(routes)))
