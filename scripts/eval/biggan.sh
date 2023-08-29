#!/bin/bash

# === Configuration ===
pool="BigGAN-239_4"
eps=0.2
shift_steps=16
shift_leap=1
batch_size=8
# =====================

declare -a EXPERIMENTS=("experiments/complete/BigGAN-239-ResNet-K200000-D120-eps0.1_6.0-FFJORD-reg1.0-seed0")

for exp in "${EXPERIMENTS[@]}"
do
  # Traverse latent space
  python traverse_latent_space.py -v --gif \
                                  --exp="${exp}" \
                                  --pool=${pool} \
                                  --eps=${eps} \
                                  --shift-steps=${shift_steps} \
                                  --shift-leap=${shift_leap} \
                                  --batch-size=${batch_size}
done
