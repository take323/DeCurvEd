#!/bin/bash

# === Configuration ===
pool="SNGAN_AnimeFaces_6"
eps=0.25
shift_steps=24
shift_leap=1
# =====================

declare -a EXPERIMENTS=("experiments/complete/SNGAN_AnimeFaces-LeNet-K200000-D128-eps0.1_6.0-FFJORD-reg1.0-seed0")

for exp in "${EXPERIMENTS[@]}"
do
  # Traverse latent space
  python traverse_latent_space.py -v --gif \
                                  --exp="${exp}" \
                                  --pool=${pool} \
                                  --eps=${eps} \
                                  --shift-steps=${shift_steps} \
                                  --shift-leap=${shift_leap}
done
