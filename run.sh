#!/usr/bin/env bash

export PYTHONPATH="$PYTHONPATH:/opt/G4Splat"

python custom/custom_refine_by_flux.py \
    --base_dir /data/liuzack0505/drone/Park \
    --output_dir /data/liuzack0505/drone/Park/output \
    --prompt "A drone shot of an empty park, featuring lanterns and art installations." \
    --data_factor 4 \
    --strength 0.3 \
    --rasterize_bg_color black

python custom/custom_refine_by_flux.py \
    --base_dir /data/liuzack0505/drone/Park2 \
    --output_dir /data/liuzack0505/drone/Park2/output \
    --prompt "An aerial view of a deserted park, complete with outdoor facilities and a playground." \
    --data_factor 4 \
    --strength 0.3 \
    --rasterize_bg_color black


python custom/custom_refine_by_flux.py \
    --base_dir /data/liuzack0505/drone/Shalun \
    --output_dir /data/liuzack0505/drone/Shalun/output \
    --prompt "A drone's-eye view capturing a high-speed rail station alongside nearby grasslands, trees, roadways, parking areas, and buildings." \
    --data_factor 4 \
    --strength 0.3 \
    --rasterize_bg_color black

python custom/custom_refine_by_flux.py \
    --base_dir /data/liuzack0505/drone/Yualin \
    --output_dir /data/liuzack0505/drone/Yualin/output \
    --prompt "A drone view tracking along the railway, bordered by nearby buildings, grassy fields, trees, and roads." \
    --data_factor 4 \
    --strength 0.3 \
    --rasterize_bg_color black

python custom/custom_refine_by_flux.py \
    --base_dir /data/liuzack0505/mipNerf360/flowers/ \
    --output_dir /data/liuzack0505/mipNerf360/flowers/output \
    --prompt "A vivid, eye-level photograph of a fiercely competitive circular flower bed in a lush green park." \
    --data_factor 4 \
    --rasterize_bg_color white

python custom/custom_refine_by_flux.py \
    --base_dir /data/liuzack0505/mipNerf360/treehill/ \
    --output_dir /data/liuzack0505/mipNerf360/treehill/output \
    --prompt "A slightly gloomy, eye-level photograph of a paved park observation area." \
    --data_factor 4 \
    --rasterize_bg_color white
