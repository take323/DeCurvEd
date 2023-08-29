#!/bin/bash

# === Configuration ===
pool="StyleGAN2_12"
eps=0.15
shift_steps=20
shift_leap=1
batch_size=2
gif_size=256
num_imgs=5
metric="corr"
# =====================

# Define experiment directories list
declare -a EXPERIMENTS=("experiments/complete/StyleGAN2-1024-W-ResNet-K200000-D512-eps0.1_6.0-FFJORD-reg1.0-smalldim-seed0")

# Define attribute groups (see `rank_interpretable_paths.py`)
declare -a ATTRIBUTE_GROUPS=("all")

# Traverse latent and attribute spaces, and rank interpretable paths for the given experiments
for exp in "${EXPERIMENTS[@]}"
do
  # --- Traverse latent space ---------------------------------------------------------------------------------------- #
  python traverse_latent_space.py -v --gif \
                                  --exp="${exp}" \
                                  --pool=${pool} \
                                  --eps=${eps} \
                                  --shift-steps=${shift_steps} \
                                  --shift-leap=${shift_leap} \
                                  --batch-size=${batch_size}
  # ------------------------------------------------------------------------------------------------------------------ #

  # --- Traverse attribute space ------------------------------------------------------------------------------------- #
  python traverse_attribute_space.py -v \
                                     --exp="${exp}" \
                                     --pool=${pool} \
                                     --eps=${eps} \
                                     --shift-steps=${shift_steps}
  # ------------------------------------------------------------------------------------------------------------------ #

  # --- Rank interpretable paths for all given attribute groups ------------------------------------------------------ #
  for attr_group in "${ATTRIBUTE_GROUPS[@]}"
  do
    python rank_interpretable_paths.py -v --exp="${exp}" \
                                          --pool=${pool} \
                                          --eps=${eps} \
                                          --shift-steps=${shift_steps} \
                                          --num-imgs=${num_imgs} \
                                          --gif-size=${gif_size} \
                                          --attr-group="${attr_group}" \
                                          --metric=${metric}
  done
  # ------------------------------------------------------------------------------------------------------------------ #
done
