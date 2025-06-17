import os
import copy
import re
import io
import logging
import json
import numpy as np
import torch
import torch.utils.data as data
from torchvision import transforms
from PIL import Image
from .base_io_dataset import BaseIODataset
from .heatmap_utils import generate_heatmap, generate_future_waypoints
from .det_utils import generate_det_data
from skimage.measure import block_reduce
from .augmenter import augment

# Taken from the LMDrive Project (OpenDILab)
# Modifications copyirght 2025 Intelligent Vehicles Lab

_logger = logging.getLogger(__name__)
_logger = logging.getLogger("train")

def lidar_to_histogram_features(lidar, crop=256):
    """
    Convert LiDAR point cloud into 2-bin histogram over 256x256 grid
    """

    def splat_points(point_cloud):
        # 256 x 256 grid
        pixels_per_meter = 8
        hist_max_per_pixel = 5
        x_meters_max = 14
        y_meters_max = 28
        xbins = np.linspace(
            -2 * x_meters_max,
            2 * x_meters_max + 1,
            2 * x_meters_max * pixels_per_meter + 1,
        )
        ybins = np.linspace(-y_meters_max, 0, y_meters_max * pixels_per_meter + 1)
        hist = np.histogramdd(point_cloud[..., :2], bins=(xbins, ybins))[0]
        hist[hist > hist_max_per_pixel] = hist_max_per_pixel
        overhead_splat = hist / hist_max_per_pixel
        return overhead_splat

    below = lidar[lidar[..., 2] <= -2.0]
    above = lidar[lidar[..., 2] > -2.0]
    below_features = splat_points(below)
    above_features = splat_points(above)
    total_features = below_features + above_features
    features = np.stack([below_features, above_features, total_features], axis=-1)
    features = np.transpose(features, (2, 0, 1)).astype(np.float32)
    return features

def transform_2d_points(xyz, r1, t1_x, t1_y, r2, t2_x, t2_y):
    """
    Build a rotation matrix and take the dot product.
    """
    # z value to 1 for rotation
    xy1 = xyz.copy()
    xy1[:, 2] = 1

    c, s = np.cos(r1), np.sin(r1)
    r1_to_world = np.matrix([[c, s, t1_x], [-s, c, t1_y], [0, 0, 1]])

    # np.dot converts to a matrix, so we explicitly change it back to an array
    world = np.asarray(r1_to_world @ xy1.T)

    c, s = np.cos(r2), np.sin(r2)
    r2_to_world = np.matrix([[c, s, t2_x], [-s, c, t2_y], [0, 0, 1]])
    world_to_r2 = np.linalg.inv(r2_to_world)

    out = np.asarray(world_to_r2 @ world).T
    # reset z-coordinate
    out[:, 2] = xyz[:, 2]

    return out


def check_data(data, info):
    for key in data:
        if isinstance(data[key], np.ndarray):
            if np.isnan(data[key]).any():
                print(key)
                print(info)
                data[key][np.isnan(data[key])] = 0
            if np.isinf(data[key]).any():
                print(key)
                print(info)
                data[key][np.isinf(data[key])] = 0
        elif isinstance(data[key], torch.Tensor):
            if torch.isnan(data[key]).any():
                print(key)
                print(info)
                data[key][torch.isnan(data[key])] = 0
            if torch.isinf(data[key]).any():
                print(key)
                print(info)
                data[key][torch.isinf(data[key])] = 0
    return data
class CarlaMVDetDataset(BaseIODataset):
    def __init__(
        self,
        root,
        towns,
        weathers,
        head="det",
        input_rgb_size=224,
        input_lidar_size=224,
        rgb_transform=None,
        depth_transform=None,
        seg_transform=None,
        lidar_transform=None,
        multi_view_transform=None,
        with_waypoints=False,
        with_seg=False,
        with_depth=False,
        with_lidar=False,
        multi_view=False,
        augment_prob=0.0,
    ):
        super().__init__()

        self.head = head
        self.input_lidar_size = input_lidar_size
        self.input_rgb_size = input_rgb_size
        self.rgb_transform = rgb_transform
        self.seg_transform = seg_transform
        self.depth_transform = depth_transform
        self.lidar_transform = lidar_transform
        self.multi_view_transform = multi_view_transform

        self.with_waypoints = with_waypoints

        self.with_seg = with_seg
        self.with_depth = with_depth
        self.with_lidar = with_lidar
        self.multi_view = multi_view
        self.augment_prob = augment_prob
        if self.augment_prob > 0:
            self.augmenter = augment(self.augment_prob)
        route_dirs = []
        self.route_frames = []
        if self.rgb_transform is None:
            self.rgb_transform = torch.from_numpy

        if isinstance(root, str):
            roots = [root]
        else:
            roots = root

        for root_dir in roots:
            if not os.path.exists(root_dir):
                _logger.warning(f"Directory not found: {root_dir}")
                continue
                
            _logger.info(f"Processing root: {root_dir}")
            
            if os.path.basename(root_dir).startswith("sub-"):
                data_dirs = [root_dir]
            else:
                data_dirs = []
                for d in os.listdir(root_dir):
                    if d.startswith("sub-"):
                        full_path = os.path.join(root_dir, d)
                        if os.path.isdir(full_path):
                            data_dirs.append(full_path)
        
            for data_dir in data_dirs:
                index_path = os.path.join(data_dir, 'dataset_index.txt')
                if not os.path.exists(index_path):
                    _logger.warning(f"No index file in: {index_path}")
                    continue
                    
                dataset_indexs = self._load_text(index_path).split('\n')
                pattern = re.compile('town(\d\d)')
                for line in dataset_indexs:
                    if len(line.split()) != 2:
                        continue
                    path, frames = line.split()
                    path = os.path.join(data_dir, 'data', path)
                    frames = int(frames)
                    _logger.info("Frames: {}".format(frames))
                    res = pattern.findall(path)
                    if len(res) != 1:
                        continue
         
                    town = int(res[0])
                    if town not in towns:
                        continue
                    for i in range(0,frames):
                        self.route_frames.append((path, i, root))
                

        _logger.info("Sub route dir nums: %d" % len(self.route_frames))


    def __len__(self):
        return len(self.route_frames)

    def __getitem__(self, idx):
        data = {}
        route_dir, frame_id, root_path = self.route_frames[idx]

        if "sub" not in route_dir:
            return self._extract_data_item_BEVDriver(route_dir,frame_id)

        rgb_image = self._load_image(
            os.path.join(route_dir, "rgb_front", "%04d.jpg" % frame_id)
        )
        rgb_left_image = self._load_image(
            os.path.join(route_dir, "rgb_left", "%04d.jpg" % frame_id)
        )
        rgb_right_image = self._load_image(
            os.path.join(route_dir, "rgb_right", "%04d.jpg" % frame_id)
        )

        if self.with_seg:
            seg_image = self._load_image(
                os.path.join(route_dir, "seg_front", "%04d.png" % frame_id)
            )
            if self.multi_view:
                seg_left_image = self._load_image(
                    os.path.join(route_dir, "seg_left", "%04d.png" % frame_id)
                )
                seg_right_image = self._load_image(
                    os.path.join(route_dir, "seg_right", "%04d.png" % frame_id)
                )
        if self.with_depth:
            depth_image = self._load_image(
                os.path.join(route_dir, "depth_front", "%04d.jpg" % frame_id)
            )
            if self.multi_view:
                depth_left_image = self._load_image(
                    os.path.join(route_dir, "depth_left", "%04d.jpg" % frame_id)
                )
                depth_right_image = self._load_image(
                    os.path.join(route_dir, "depth_right", "%04d.jpg" % frame_id)
                )

        '''
        You can use tools/data/batch_merge_data.py to generate FULL measurements for reducing io cost
        measurements = self._load_json(
            os.path.join(route_dir, "measurements_full", "%04d.json" % frame_id)
        )
        actors_data = measurements["actors_data"]
        stop_sign = int(measurements["stop_sign"])
        '''

        measurements = self._load_json(
            os.path.join(route_dir, "measurements", "%04d.json" % frame_id)
        )
        actors_data = self._load_json(
            os.path.join(route_dir, "actors_data", "%04d.json" % frame_id)
        )
      

        if len(measurements['is_red_light_present']) > 0:
            traffic_light_state = 0
        else:
            traffic_light_state = 1
 
        if self.with_lidar:
            lidar_unprocessed = self._load_npy(
                os.path.join(route_dir, "lidar", "%04d.npy" % frame_id)
            )[..., :3]
            lidar_unprocessed[:, 1] *= -1
            full_lidar = transform_2d_points(
                lidar_unprocessed,
                np.pi / 2 - measurements["theta"],
                -measurements["gps_x"],
                -measurements["gps_y"],
                np.pi / 2 - measurements["theta"],
                -measurements["gps_x"],
                -measurements["gps_y"],
            )
            lidar_processed = lidar_to_histogram_features(
                full_lidar, crop=self.input_lidar_size
            )

        cmd_one_hot = [0, 0, 0, 0, 0, 0]
        cmd = measurements["command"] - 1
        if cmd < 0:
            cmd = 3
        cmd_one_hot[cmd] = 1
        cmd_one_hot.append(measurements["speed"])
        mes = np.array(cmd_one_hot)
        mes = torch.from_numpy(mes).float()

        data["measurements"] = mes
        #data['command'] = torch.from_numpy(np.array(cmd))

        if np.isnan(measurements["theta"]):
            measurements["theta"] = 0
        ego_theta = measurements["theta"]
        x_command = measurements["x_command"]
        y_command = measurements["y_command"]
        if "gps_x" in measurements:
            ego_x = measurements["gps_x"]
        else:
            ego_x = measurements["x"]
        if "gps_y" in measurements:
            ego_y = measurements["gps_y"]
        else:
            ego_y = measurements["y"]
        R = np.array(
            [
                [np.cos(np.pi / 2 + ego_theta), -np.sin(np.pi / 2 + ego_theta)],
                [np.sin(np.pi / 2 + ego_theta), np.cos(np.pi / 2 + ego_theta)],
            ]
        )
        local_command_point = np.array([x_command - ego_x, y_command - ego_y])
        local_command_point = R.T.dot(local_command_point)
        if any(np.isnan(local_command_point)):
            local_command_point[np.isnan(local_command_point)] = np.mean(
                local_command_point
            )
        local_command_point = torch.from_numpy(local_command_point).float()
        data["target_point"] = local_command_point


        if self.rgb_transform is not None:
            rgb_main_image = self.rgb_transform(rgb_image)
        data["rgb"] = rgb_main_image

        if self.rgb_center_transform is not None:
            rgb_center_image = self.rgb_center_transform(rgb_image)
        data["rgb_center"] = rgb_center_image

        if self.with_seg:
            if self.seg_transform is not None:
                seg_image = self.seg_transform(seg_image)
            data_seg = seg_image


        if self.with_depth:
            if self.depth_transform is not None:
                depth_image = self.depth_transform(depth_image)
            data["depth"] = depth_image
            if self.multi_view:
                if self.multi_view_transform is not None:
                    depth_left_image = self.multi_view_transform(depth_left_image)
                    depth_right_image = self.multi_view_transform(depth_right_image)
                data["depth_left"] = depth_left_image
                data["depth_right"] = depth_right_image

        if self.with_lidar:
            if self.lidar_transform is not None:
                lidar_processed = self.lidar_transform(lidar_processed)
            data["lidar"] = lidar_processed
        if self.multi_view:
            if self.multi_view_transform is not None:
                rgb_left_image = self.multi_view_transform(rgb_left_image)
                rgb_right_image = self.multi_view_transform(rgb_right_image)
            data["rgb_left"] = rgb_left_image
            data["rgb_right"] = rgb_right_image

        heatmap = generate_heatmap(
            copy.deepcopy(measurements), copy.deepcopy(actors_data)
        )
        det_data = (
            generate_det_data(
                heatmap, copy.deepcopy(measurements), copy.deepcopy(actors_data)
            )
            .reshape(400, -1)
            .astype(np.float32)
        )
        img_det_traffic = heatmap[:100, 40:140, None]
        img_det_traffic = transforms.ToTensor()(img_det_traffic)


        return data, (
            img_det_traffic,
            det_data, 
            data_seg,
            1,
            traffic_light_state
        )


    def _extract_data_item_BEVDriver(self, route_dir, frame_id, with_actor_infos=False, cached_measurements=None):
        data = {}
        # You can use tools/data/batch_merge_data.py to generate FULL image (including front, left, right) for reducing io cost
        rgb_full_image = self._load_image(
            os.path.join(route_dir, "rgb_full", "%04d.jpg" % frame_id)
        )

        rgb_image = rgb_full_image.crop((0, 0, 800, 600))
        rgb_left_image = rgb_full_image.crop((0, 600, 800, 1200))
        rgb_right_image = rgb_full_image.crop((0, 1200, 800, 1800))
        #rgb_rear_image = rgb_full_image.crop((0, 1800, 800, 2400))


        if cached_measurements:
            measurements = cached_measurements
        else:
            measurements_all = self._load_json(
                os.path.join(route_dir, "measurements_all.json")
            )
            measurements = self._load_json(
                os.path.join(route_dir, "measurements_full", "%04d.json" % frame_id)
            )
        actors_data = measurements["actors_data"]
)
        '''
        # You can use tools/data/batch_merge_data.py to generate FULL measurements for reducing io cost
        measurements = self._load_json(
            os.path.join(route_dir, "measurements", "%04d.json" % frame_id)
        )
        actors_data = self._load_json(
            os.path.join(route_dir, "actors_data", "%04d.json" % frame_id)
        )
        affordances = self._load_npy(os.path.join(route_dir, 'affordances/%04d.npy' % frame_id))
        stop_sign = int(affordances.item()['stop_sign'])
        '''

        if self.with_lidar:
            lidar_unprocessed = self._load_npy(
                os.path.join(route_dir, "lidar", "%04d.npy" % frame_id)
            )[..., :3]
            lidar_unprocessed[:, 1] *= -1
            full_lidar = transform_2d_points(
                lidar_unprocessed,
                np.pi / 2 - measurements["theta"],
                -measurements["gps_x"],
                -measurements["gps_y"],
                np.pi / 2 - measurements["theta"],
                -measurements["gps_x"],
                -measurements["gps_y"],
                )
            lidar_processed = lidar_to_histogram_features(
                full_lidar, crop=self.input_lidar_size
            )

        if self.with_lidar:
            if self.lidar_transform is not None:
                lidar_processed = self.lidar_transform(lidar_processed)
            data["lidar"] = lidar_processed
        cmd_one_hot = [0, 0, 0, 0, 0, 0]
        cmd = measurements["command"] - 1
        if cmd < 0:
            cmd = 3
        cmd_one_hot[cmd] = 1
        cmd_one_hot.append(measurements["speed"])
        mes = np.array(cmd_one_hot)
        mes = torch.from_numpy(mes).float()

        data["measurements"] = mes
        #data['velocity'] = torch.from_numpy(np.array([measurements['speed']])).float()
        #data['command'] = torch.from_numpy(np.array(cmd))

        if np.isnan(measurements["theta"]):
            measurements["theta"] = 0
        ego_theta = measurements["theta"]
        x_command = measurements["x_command"]
        y_command = measurements["y_command"]
        ego_x = measurements["gps_x"]
        ego_y = measurements["gps_y"]
        R = np.array(
            [
                [np.cos(np.pi / 2 + ego_theta), -np.sin(np.pi / 2 + ego_theta)],
                [np.sin(np.pi / 2 + ego_theta), np.cos(np.pi / 2 + ego_theta)],
            ]
        )
        local_command_point = np.array([x_command - ego_x, y_command - ego_y])
        local_command_point = R.T.dot(local_command_point)
        if any(np.isnan(local_command_point)):
            local_command_point[np.isnan(local_command_point)] = np.mean(
                local_command_point
            )
        local_command_point = torch.from_numpy(local_command_point).float()
        data["target_point"] = local_command_point

        if self.rgb_transform is not None:
            rgb_main_image = self.rgb_transform(rgb_image)
        data["rgb"] = rgb_main_image

        if self.rgb_center_transform is not None:
            rgb_center_image = self.rgb_center_transform(rgb_image)
        data["rgb_center"] = rgb_center_image





        if self.multi_view:
            if self.multi_view_transform is not None:
                rgb_left_image = self.multi_view_transform(rgb_left_image)
                rgb_right_image = self.multi_view_transform(rgb_right_image)
                #rgb_rear_image = self.multi_view_transform(rgb_rear_image)
            data["rgb_left"] = rgb_left_image
            data["rgb_right"] = rgb_right_image
            #data["rgb_rear"] = rgb_rear_image

        heatmap = generate_heatmap(
            copy.deepcopy(measurements), copy.deepcopy(actors_data)
        )
        if with_actor_infos:
            det_data, actor_infos = (
                generate_det_data(heatmap, copy.deepcopy(measurements), copy.deepcopy(actors_data), return_object_ids=True))
            det_data = det_data.reshape(400, -1).astype(np.float32)
        else:
            det_data = (
                generate_det_data(
                    heatmap, copy.deepcopy(measurements), copy.deepcopy(actors_data)).reshape(400, -1).astype(np.float32)
            )
        '''
        heatmap_mask = np.zeros((50, 50), dtype=bool).reshape(-1)
        reduced_heatmap = block_reduce(heatmap[:250, 25:275], block_size=(5, 5), func=np.mean)
        '''
        #data['heatmap_mask'] = transforms.ToTensor()(heatmap_mask)
        
        img_traffic = heatmap[:100, 40:140, None]
        img_traffic = transforms.ToTensor()(img_traffic)
        '''
        img_traj = generate_future_waypoints(measurements)
        img_traj = img_traj[:250, 25:275, None]
        img_traj = transforms.ToTensor()(img_traj)

        reduced_heatmap = reduced_heatmap[:, :, None]
        reduced_heatmap = transforms.ToTensor()(reduced_heatmap) / 255.
        '''    
        data = check_data(data, info=route_dir+str(frame_id))

        if with_actor_infos:
            return data, (
                img_traffic,
                command_waypoints,
                is_junction,
                traffic_light_state,
                det_data,
                img_traj,
                stop_sign,
                reduced_heatmap,
            ), actor_infos


        return data, (
            img_traffic,
            det_data,
            torch.zeros((1,224,224)),
            0
        )