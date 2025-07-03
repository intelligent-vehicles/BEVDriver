# BEVDriver

## News

🎉 Our paper *"BEVDriver: Leveraging BEV Maps in LLMs for Robust Closed-Loop Driving"* has been **accepted to IROS 2025**! [Read it here](https://arxiv.org/abs/2503.03074)

🛠️ We have also **released the full codebase** — feel free to explore, reproduce, and build upon our work.

---

## Description  
We introduce: BEVDriver, an LLM-based motion planner using bird-eye-view maps for high level maneuver planning and waypoint prediction, following natural language instructions.


## Demo Video

<video src="https://private-user-images.githubusercontent.com/144470111/420561528-792e4cc7-48a9-4888-b12f-6dcc544c4819.mp4?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3NDE0MDk4MDksIm5iZiI6MTc0MTQwOTUwOSwicGF0aCI6Ii8xNDQ0NzAxMTEvNDIwNTYxNTI4LTc5MmU0Y2M3LTQ4YTktNDg4OC1iMTJmLTZkY2M1NDRjNDgxOS5tcDQ_WC1BbXotQWxnb3JpdGhtPUFXUzQtSE1BQy1TSEEyNTYmWC1BbXotQ3JlZGVudGlhbD1BS0lBVkNPRFlMU0E1M1BRSzRaQSUyRjIwMjUwMzA4JTJGdXMtZWFzdC0xJTJGczMlMkZhd3M0X3JlcXVlc3QmWC1BbXotRGF0ZT0yMDI1MDMwOFQwNDUxNDlaJlgtQW16LUV4cGlyZXM9MzAwJlgtQW16LVNpZ25hdHVyZT0yMzJlNzkwMzI1YTg0MmRjMGY0YjQyODk3MTU0M2Q2ZWYxZGViMGIwYmM3NGY1MzA3Zjc4OGQyZWYxNDBhMTgwJlgtQW16LVNpZ25lZEhlYWRlcnM9aG9zdCJ9.WqYVAEnUdARX7dglynfESKweKrw_ACoWrop2v4OX6qY" controls width="600"></video>


# Code

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)


## About this Repo
This repo contains the code to run and evaluate BEVDriver in Carla. 
It should be structured as follows:

```
├── BEVDriver
│   └── carla
│   └── data_collection
│   └── LAVIS
│   └── leaderboard
│   └── scenario_runner
│   └── timm
│   └── tools
│   └── BEV_encoder
│   └── ...
│   │    └── train.sh
│   │    └── ...
│   └── ...
├── README.md
├── .gitignore
```

## Setup this repo 

To set up the BEVDriver project, clone the BEVDriver repo: `

```
git clone https://github.com/intelligent-vehicles/bevdriver
cd BEVDriver
```

Create a conda environment using python version 3.8.

```
conda create -n bevdriver python=3.8
conda activate bevdriver
```

Inside your environment, install the required packages:

```
pip install -r requirements.txt
cd LAVIS
pip install -r requirements.txt
cd ..
```
Note: Do not install timm or LAVIS. Use the adapted versions included in the Git Repo.

## Set up Carla 0.9.10.1

Run the following code for Carla download and setup:

```
chmod +x setup_carla.sh
./setup_carla.sh
pip install carla
```


## Data Collection

You can find an existing dataset published by the OpenDILab for `LMDrive` [here](https://huggingface.co/datasets/OpenDILabCommunity/LMDrive).

We modified the data collection pipeline from the OpenDILab.
To generate your own data with semantic segmentation, follow these steps:

Run the following code to generate subfolders with parallelizable bash scripts on 4 servers (modify for fewer servers):

```
cd BEVDriver/dataset
python init_dir.py 
cd ../data_collection
python generate_bashs.py
python generate_batch_collect.py 
cd..
```


To collect data (including semantic segmentation) on a specific town, run the auto-pilot as expert-driver with

``` 
bash data_collection/batch_run/run_route_routes_town01_long.sh
```
and replace `run_route_routes_town01_long.sh` with your desired script, or use our collection script to collect data on various town with 4 carla servers:

```
./data_collection/run_batch_collect.sh
```

Uncomment the required routes.

After data collection, to make the dataset ready for processing (necessary for full pipeline training), follow these preprocessing steps, which are adapted to the BEVDriver datastructure:

```
python3 tools/data_preprocessing/index_routes.py dataset # generates a list of existing routes
python3 tools/data_preprocessing/batch_stat_blocked_data.py dataset # find long frame series where vehicle is blocked 
python3 tools/data_preprocessing/batch_rm_blocked_data.py dataset # remove these frames with blocked vehicle
python3 tools/data_preprocessing/batch_recollect_data.py dataset # reorganize frame ids
python3 tools/data_preprocessing/batch_merge_measurements.py dataset # merge measurements into one file for reduced access time
python3 tools/data_preprocessing/batch_merge_data.py dataset # merge rgb images into rgb_full to reduce access time
python3 tools/data_parsing/parse_instruction.py dataset # generates navigation instructions and saves them to navigation_instruction_list.txt
python3 tools/data_parsing/parse_misleading.py dataset # generates misleading navigation instructions and saves them to misleading_data.txt
```

The data will be saved to a 'dataset' folder, in which the sub-folders have been generated in the previous step. 


## Train the BEV-Encoder

You find the encoder model for training in `timm/models/bevdriver_encoder-train.py`. This model includes both encoder and decoder. 

For training on the collected data, start the distributed training script:
```
./BEV_encoder/train.sh
```


## Run the whole model pipeline

To train the full model, download the base LLM (e.g. [Llama-7b](https://huggingface.co/huggyllama/llama-7b)) and adjust the paths to the encoder and model checkpoints and dataset.

``` 
cd LAVIS
./run.sh {num_gpus} lavis/projects/bevdriver/train_modular.yaml
```


## Checkpoints

Find the pretrained Checkpoints from our paper here: 


| Checkpoint | Description|
|---------|------------------|
| [BEVDriver Llama-7b](https://syncandshare.lrz.de/getlink/fijcZ1H9GEXafEyQjBKUf/BEV%20Encoder%20with%20Traffic%20Light%20Loss) | Model (Trained on frozen BEV Encoder)|
| [BEV Encoder](https://syncandshare.lrz.de/getlink/fiWRzThZRF4xY6DN2Ets7/Main%20Model%20Llama-7b)     | BEV Encoder incl. Traffic Light Status | 


## Evaluation 

For evaluation, enter the correct paths for the model checkpoints and base LLM in `leaderboard/team_code/bevdriver_config.py`.

```
./leaderboard/scripts/run_evaluation.sh
```

Adjust the `run_evaluation.sh` script to different routes and scenarios, e.g. LangAuto Tiny:

```
export SCENARIOS=leaderboard/data/LangAuto/tiny.json
export ROUTES=leaderboard/data/LangAuto/tiny.xml
``` 

## Citation

If you use this work in your research, please cite us using the following BibTeX entry:

```
@article{winter2025bevdriver,
  title={BEVDriver: Leveraging BEV Maps in LLMs for Robust Closed-Loop Driving},
  author={Winter, Katharina and Azer, Mark and Flohr, Fabian B},
  journal={arXiv preprint arXiv:2503.03074},
  year={2025}
}
```

## Acknowledgements

- [LMDrive](https://github.com/opendilab/LMDrive)
- [InterFuser](https://github.com/opendilab/InterFuser)
- [CARLA Leaderboard](https://github.com/carla-simulator/leaderboard)
- [CARLA Challenge](https://github.com/bradyz/2020_CARLA_challenge)
- [Scenario Runner](https://github.com/carla-simulator/scenario_runner)
- [LAVIS](https://github.com/salesforce/LAVIS)
- [Pytorch Image Models](https://github.com/huggingface/pytorch-image-models)
- [PSP Net](https://arxiv.org/pdf/1612.01105)


## License
All code within this repository is under the [Apache 2.0 License](https://www.apache.org/licenses/LICENSE-2.0)
