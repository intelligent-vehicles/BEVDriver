#!/usr/bin/env bash
# Download and install CARLA

# This file is part of the BEVDriver project:
# https://github.com/intelligent-vehicles/bevdriver
#
# Taken from the LMDrive Project:
# https://github.com/opendilab/LMDrive
# 
# Copyright [original year] OpenDILab contributors.
# Licensed under the Apache License, Version 2.0.
# You may obtain a copy of the license at:
# http://www.apache.org/licenses/LICENSE-2.0
#
# This file has been copied without modification.


mkdir carla
cd carla
wget https://carla-releases.s3.us-east-005.backblazeb2.com/Linux/CARLA_0.9.10.1.tar.gz
wget https://carla-releases.s3.us-east-005.backblazeb2.com/Linux/AdditionalMaps_0.9.10.1.tar.gz
tar -xf CARLA_0.9.10.1.tar.gz
tar -xf AdditionalMaps_0.9.10.1.tar.gz
rm CARLA_0.9.10.1.tar.gz
rm AdditionalMaps_0.9.10.1.tar.gz
cd ..
