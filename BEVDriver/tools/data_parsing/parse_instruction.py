import os
import re
import sys
import json
from tqdm import tqdm

from turn_rules import Turn01, Turn02, Turn03, Turn04, Turn05, Turn06
from follow_rules import Follow01, Follow02, Follow03, Follow04
from other_rules import Other01, Other02, Other03, Other04, Other05

registered_class = {
    'Turn-01-L': Turn01(direction='left'),
    'Turn-01-R': Turn01(direction='right'),
    'Turn-01-L-dis': Turn01(direction='left', dis=True),
    'Turn-01-R-dis': Turn01(direction='right', dis=True),
    'Turn-02-L': Turn02(direction='left'),
    'Turn-02-R': Turn02(direction='right'),
    'Turn-02-S': Turn02(direction='straight'),
    'Turn-02-L-dis': Turn02(direction='left', dis=True),
    'Turn-02-R-dis': Turn02(direction='right', dis=True),
    'Turn-02-S-dis': Turn02(direction='straight', dis=True),
    'Turn-03-L': Turn03(direction='left'),
    'Turn-03-R': Turn03(direction='right'),
    'Turn-03-S': Turn03(direction='straight'),
    'Turn-03-L-dis': Turn03(direction='left', dis=True),
    'Turn-03-R-dis': Turn03(direction='right', dis=True),
    'Turn-03-S-dis': Turn03(direction='straight', dis=True),
    'Turn-04-L': Turn04(direction='left'),
    'Turn-04-R': Turn04(direction='right'),
    'Turn-04-S': Turn04(direction='straight'),
    'Turn-04-L-dis': Turn04(direction='left', dis=True),
    'Turn-04-R-dis': Turn04(direction='right', dis=True),
    'Turn-04-S-dis': Turn04(direction='straight', dis=True),
    'Turn-05-1': Turn05(exit_no=1),
    'Turn-05-2': Turn05(exit_no=2),
    'Turn-05-3': Turn05(exit_no=3),
    'Turn-06-L-L': Turn06(first_direction='left', second_direction='left'),
    'Turn-06-L-R': Turn06(first_direction='left', second_direction='right'),
    'Turn-06-L-S': Turn06(first_direction='left', second_direction='straight'),
    'Turn-06-R-L': Turn06(first_direction='right', second_direction='left'),
    'Turn-06-R-R': Turn06(first_direction='right', second_direction='right'),
    'Turn-06-R-S': Turn06(first_direction='right', second_direction='straight'),
    'Turn-06-S-L': Turn06(first_direction='straight', second_direction='left'),
    'Turn-06-S-R': Turn06(first_direction='straight', second_direction='right'),
    'Turn-06-S-S': Turn06(first_direction='straight', second_direction='straight'),
    'Follow-01-L': Follow01(direction='left'),
    'Follow-01-R': Follow01(direction='right'),
    'Follow-01-L-dis': Follow01(direction='left', dis=True),
    'Follow-01-R-dis': Follow01(direction='right', dis=True),
    'Follow-02-s1': Follow02(style=1),
    'Follow-02-s2': Follow02(style=2),
    'Follow-02-s1-dis': Follow02(style=1, dis=True),
    'Follow-02-s2-dis': Follow02(style=2, dis=True),
    'Follow-03-s1': Follow03(style=1),
    'Follow-03-s2': Follow03(style=2),
    'Follow-03-s1-dis': Follow03(style=1, dis=True),
    'Follow-03-s2-dis': Follow03(style=2, dis=True),
    'Follow-04-L': Follow04(directions=['left','straight']),
    'Follow-04-R': Follow04(directions=['right']),
    'Follow-04-L-dis': Follow04(directions=['left','straight'], dis=True),
    'Follow-04-R-dis': Follow04(directions=['right'], dis=True),
    'Other-02': Other02(),
    'Other-03': Other03(),
    'Other-04': Other04(),
    'Other-05': Other05(),
}

rule_id_mapping_dict = {k: i for i, k in enumerate(registered_class)}
processing_rules = list(rule_id_mapping_dict.keys())

def process(line, dataset_root):
    try:
        processed_data = []
        path, frames = line.split()
        frames = int(frames.strip())

        sub_dir = path.split(os.sep)[0]  
        dir_path = os.path.join(dataset_root, path)

        if not os.path.isdir(dir_path):
            raise FileNotFoundError(f"Route path does not exist: {dir_path}")

        town_id = int(re.findall(r'town(\d\d)', dir_path)[0])
        weather_id = int(re.findall(r'_w(\d+)_', dir_path)[0]) if '_w' in dir_path else 0

        json_data = []
        for frame_id in range(frames):
            measurement_path = os.path.join(dir_path, 'measurements', f"{frame_id:04d}.json")
            actor_path = os.path.join(dir_path, 'actors_data', f"{frame_id:04d}.json")
            if not os.path.exists(measurement_path):
                continue
            with open(measurement_path, 'r') as f:
                frame_data = json.load(f)
            if os.path.exists(actor_path):
                with open(actor_path, 'r') as fa:
                    frame_data["actors_data"] = json.load(fa)
            json_data.append(frame_data)

        if not json_data:
            return []

        for rule in processing_rules:
            results = registered_class[rule].process({
                'data': json_data,
                'town_id': town_id,
                'weather_id': weather_id
            })
            rule_id = rule_id_mapping_dict[rule]
            for result in results:
                result.update({
                    'instruction': rule,
                    'instruction_id': rule_id,
                    'town_id': town_id,
                    'weather_id': weather_id,
                    'route_path': path,
                    'route_frames': frames,
                    'bad_case': 'False'
                })
                processed_data.append(result)

        return processed_data

    except Exception as e:
        exc_type, exc_val, exc_tb = sys.exc_info()
        exc_file = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print({
            "Error type: ": exc_type,
            "Error information": exc_val,
            "Error file": exc_file,
            "Error line": exc_tb.tb_lineno
        })
        return []

if __name__ == '__main__':
    dataset_root = sys.argv[1]
    list_file = os.path.join(dataset_root, 'dataset_index.txt')
    lines = open(list_file, 'r').readlines()

    results_per_line = []
    for line in tqdm(lines):
        result = process(line.strip(), dataset_root)
        results_per_line.append(result)

    with open(os.path.join(dataset_root, 'navigation_instruction_list.txt'), 'w') as f_write:
        for results in results_per_line:
            if not results:
                continue
            for result in results:
                f_write.write(json.dumps(result) + '\n')
