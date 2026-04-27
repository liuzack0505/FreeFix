#!/usr/bin/env bash

export PYTHONPATH="$PYTHONPATH:/opt/G4Splat"

python custom/custom_refine_by_flux.py \
    --base_dir /data/liuzack0505/drone/Park \
    --output_dir /data/liuzack0505/drone/Park/output \
    --prompt "A drone shot of an empty park, featuring lanterns and art installations."

python custom/custom_refine_by_flux.py \
    --base_dir /data/liuzack0505/drone/Park2 \
    --output_dir /data/liuzack0505/drone/Park2/output \
    --prompt "An aerial view of a deserted park, complete with outdoor facilities and a playground."

python custom/custom_refine_by_flux.py \
    --base_dir /data/liuzack0505/drone/Shalun \
    --output_dir /data/liuzack0505/drone/Shalun/output \
    --prompt "A drone's-eye view capturing a high-speed rail station alongside nearby grasslands, trees, roadways, parking areas, and buildings."

python custom/custom_refine_by_flux.py \
    --base_dir /data/liuzack0505/drone/Park2 \
    --output_dir /data/liuzack0505/drone/Park2/output \
    --prompt "A drone view tracking along the railway, bordered by nearby buildings, grassy fields, trees, and roads."
