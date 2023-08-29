#!/bin/bash


# === Experiment configuration ===
gan_type="SNGAN_MNIST"
learn_alphas=false
learn_gammas=false
num_support_sets=200000
num_support_dipoles=128
min_shift_magnitude=0.1
max_shift_magnitude=6.0
reconstructor_type="LeNet"
batch_size=128
max_iter=200000
tensorboard=true
lambda_reg=0.25
flow="FFJORD"
reg_flow=1.0
small_dim=""
kld=""
temp=1
flow_modules='128-128-128-128-128'
# ================================


# Run training script
learn_a=""
if $learn_alphas ; then
  learn_a="--learn-alphas"
fi

learn_g=""
if $learn_gammas ; then
  learn_g="--learn-gammas"
fi

tb=""
if $tensorboard ; then
  tb="--tensorboard"
fi

python train.py $tb \
                --gan-type=${gan_type} \
                --reconstructor-type=${reconstructor_type} \
                $learn_a \
                $learn_g \
                --num-support-sets=${num_support_sets} \
                --num-support-dipoles=${num_support_dipoles} \
                --min-shift-magnitude=${min_shift_magnitude} \
                --max-shift-magnitude=${max_shift_magnitude} \
                --batch-size=${batch_size} \
                --max-iter=${max_iter} \
                --log-freq=100 \
                --ckp-freq=50000 \
		            --lambda-reg=${lambda_reg} \
                --flow=${flow} \
		            --reg-flow=${reg_flow} \
                $small_dim \
                --flow-modules=${flow_modules}
