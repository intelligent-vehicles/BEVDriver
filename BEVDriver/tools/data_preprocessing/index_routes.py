import os
import sys
import glob
from tqdm import tqdm

if __name__ == "__main__":
    dataset_root = sys.argv[1]

    data_path = os.path.join(dataset_root, "sub-*/data")
    data_dirs = glob.glob(data_path)

    all_entries = []

    for data_dir in data_dirs:
        sub_dir = os.path.dirname(data_dir)
        list_file = os.path.join(sub_dir, "dataset_index.txt")

        routes = os.listdir(data_dir)
        with open(list_file, 'w') as f:
            for route in tqdm(routes, desc=f"Indexing {os.path.basename(sub_dir)}"):
                route_path = os.path.join(data_dir, route)
                measurements_path = os.path.join(route_path, 'measurements')

                if os.path.isdir(route_path) and os.path.isdir(measurements_path):
                    frames = len(os.listdir(measurements_path))
                    if frames < 32:
                        print(f"Route {route_path} only has {frames} frames (<32). Omitted.")
                    else:
                        # Write just the route name and frame count to sub-dir file
                        f.write(f"{route} {frames}\n")

                        # Full relative path for the combined index
                        rel_path = os.path.relpath(route_path, dataset_root)
                        all_entries.append(f"{rel_path} {frames}")

    # Write the merged dataset_index.txt in dataset root
    combined_index_file = os.path.join(dataset_root, "dataset_index.txt")
    with open(combined_index_file, 'w') as f:
        for entry in all_entries:
            f.write(entry + '\n')
