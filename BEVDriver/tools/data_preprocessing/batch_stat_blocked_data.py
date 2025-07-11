import os
import json
import sys
import glob

import numpy as np
from tqdm import tqdm
from multiprocessing import Pool


def process(route_dir):
    res = []
    try:
        frames = len(os.listdir(os.path.join(route_dir, "measurements")))
        stop = 0
        max_stop = 0
        last_actors_num = 0
        for i in range(frames):
            json_data = json.load(
                open(os.path.join(route_dir, "measurements", "%04d.json" % i))
            )
            actors_data = json.load(
                open(os.path.join(route_dir, "actors_data", "%04d.json" % i))
            )
            actors_num = len(actors_data)
            light = json_data["is_red_light_present"]
            speed = json_data["speed"]
            junction = json_data["is_junction"]
            brake = json_data["should_brake"]
            if speed < 0.1 and len(light) == 0 and brake == 1:
                stop += 1
                max_stop = max(max_stop, stop)
            else:
                if stop >= 10 and actors_num < last_actors_num:
                    print('Find route: %s blocked from %d to %d' % (route_dir, i, i + stop))
                    res.append((route_dir, i, stop))
                stop = 0
            last_actors_num = actors_num
        if stop >= 10:
            print('Find route: %s blocked from %d to %d' % (route_dir, i, i + stop))
            res.append((route_dir, frames - 1, stop))
    except Exception as e:
        print(e)
        print('The folder %s has an existing problem' % route_dir)
    return res


def find_all_routes(dataset_root):
    routes = []

    # Option 1: Direct route folders under dataset
    for route_dir in glob.glob(os.path.join(dataset_root, "routes*")):
        if os.path.isdir(os.path.join(route_dir, "measurements")):
            routes.append(route_dir)

    # Option 2: Nested under sub-*/data/routes*/
    nested_pattern = os.path.join(dataset_root, "sub-*/data/routes*/measurements")
    for measurement_path in glob.glob(nested_pattern):
        print("Measurement Path: ", measurement_path)
        route_dir = os.path.dirname(measurement_path)
        routes.append(route_dir)

    return routes


if __name__ == "__main__":
    dataset_root = sys.argv[1]

    # Use dataset_index.txt if it exists
    list_file = os.path.join(dataset_root, 'dataset_index.txt')
    if os.path.isfile(list_file):
        routes = []
        for line in open(list_file, "r").readlines():
            path = line.split()[0].strip()
            routes.append(os.path.join(dataset_root, path))
    else:
        print("dataset_index.txt not found, scanning file tree for route folders...")
        routes = find_all_routes(dataset_root)

    f = open(os.path.join(dataset_root, "blocked_stat.txt"), "w")
    with Pool(8) as p:
        for t in tqdm(p.imap(process, routes), total=len(routes)):
            if len(t) == 0:
                continue
            for tt in t:
                f.write("%s %d %d\n" % tt)
    f.close()
