import argparse
import random
import numpy as np
import torch
from lib import *
from models.gan_load import build_biggan, build_proggan, build_stylegan2, build_sngan


def main():
    """WarpedGANSpace -- Training script.

    Options:
        ===[ Pre-trained GAN Generator (G) ]============================================================================
        --gan-type                 : set pre-trained GAN type
        --z-truncation             : set latent code sampling truncation parameter. If set, latent codes will be sampled
                                     from a standard Gaussian distribution truncated to the range [-args.z_truncation,
                                     +args.z_truncation]
        --biggan-target-classes    : set list of classes to use for conditional BigGAN (see BIGGAN_CLASSES in
                                     lib/config.py). E.g., --biggan-target-classes 14 239.
        --stylegan2-resolution     : set StyleGAN2 generator output images resolution:  256 or 1024 (default: 1024)
        --shift-in-w-space         : search latent paths in StyleGAN2's W-space (otherwise, look in Z-space)

        ===[ Support Sets (S) ]=========================================================================================
        -K, --num-support-sets     : set number of support sets; i.e., number of warping functions -- number of
                                     interpretable paths
        -D, --num-support-dipoles  : set number of support dipoles per support set
        --learn-alphas             : learn RBF alpha params
        --learn-gammas             : learn RBF gamma params
        -g, --gamma                : set RBF gamma param (by default, gamma will be set to the inverse of the latent
                                     space dimensionality)
        --support-set-lr           : set learning rate for learning support sets

        ===[ Reconstructor (R) ]========================================================================================
        --reconstructor-type       : set reconstructor network type
        --min-shift-magnitude      : set minimum shift magnitude
        --max-shift-magnitude      : set maximum shift magnitude
        --reconstructor-lr         : set learning rate for reconstructor R optimization

        ===[ Training ]=================================================================================================
        --max-iter                 : set maximum number of training iterations
        --batch-size               : set training batch size
        --lambda-cls               : classification loss weight
        --lambda-reg               : regression loss weight
        --log-freq                 : set number iterations per log
        --ckp-freq                 : set number iterations per checkpoint model saving
        --tensorboard              : use TensorBoard

        ===[ CUDA ]=====================================================================================================
        --cuda                     : use CUDA during training (default)
        --no-cuda                  : do NOT use CUDA during training
        ================================================================================================================
    """
    parser = argparse.ArgumentParser(description="WarpedGANSpace training script")

    # === Pre-trained GAN Generator (G) ============================================================================== #
    parser.add_argument('--gan-type', type=str, choices=GAN_WEIGHTS.keys(), help='set GAN generator model type')
    parser.add_argument('--z-truncation', type=float, help="set latent code sampling truncation parameter")
    parser.add_argument('--biggan-target-classes', nargs='+', type=int, help="list of classes for conditional BigGAN")
    parser.add_argument('--stylegan2-resolution', type=int, default=1024, choices=(64, 128, 256, 512, 1024),
                        help="StyleGAN2 image resolution")
    parser.add_argument('--shift-in-w-space', action='store_true', help="search latent paths in StyleGAN2's W-space")

    # === Support Sets (S) ======================================================================== #
    parser.add_argument('-K', '--num-support-sets', type=int, help="set number of support sets (warping functions)")
    parser.add_argument('-D', '--num-support-dipoles', type=int, help="set number of support dipoles per support set")
    parser.add_argument('--learn-alphas', action='store_true', help='learn RBF alpha params')
    parser.add_argument('--learn-gammas', action='store_true', help='learn RBF gamma params')
    parser.add_argument('-g', '--gamma', type=float, help="set RBF gamma param; when --learn-gammas is set, this will "
                                                          "be the initial value of gammas for all RBFs")
    parser.add_argument('--support-set-lr', type=float, default=1e-4, help="set learning rate")

    # === Reconstructor (R) ========================================================================================== #
    parser.add_argument('--reconstructor-type', type=str, choices=RECONSTRUCTOR_TYPES, default='ResNet',
                        help='set reconstructor network type')
    parser.add_argument('--min-shift-magnitude', type=float, default=0.25, help="set minimum shift magnitude")
    parser.add_argument('--max-shift-magnitude', type=float, default=0.45, help="set shifts magnitude scale")
    parser.add_argument('--reconstructor-lr', type=float, default=1e-4,
                        help="set learning rate for reconstructor R optimization")
    # === FLOW =============
    parser.add_argument('--flow', type=str, default='FFJORD', help='nice, realnvp, glow, FFJORD, RBF, or linear')
    parser.add_argument('--reg-flow', type=float, default=1., help='regularizer loss weight')
    parser.add_argument('--small-dim', action='store_true', help='edit part of dimensions')
    parser.add_argument("--flow-modules", type=str, default='512-512-512-512-512', help='flow modules for FFJORD')

    # === Training =================================================================================================== #
    parser.add_argument('--max-iter', type=int, default=200000, help="set maximum number of training iterations")
    parser.add_argument('--batch-size', type=int, default=32, help="set batch size")
    parser.add_argument('--lambda-cls', type=float, default=1.00, help="classification loss weight")
    parser.add_argument('--lambda-reg', type=float, default=0.25, help="regression loss weight")
    parser.add_argument('--log-freq', default=10, type=int, help='set number iterations per log')
    parser.add_argument('--ckp-freq', default=1000, type=int, help='set number iterations per checkpoint model saving')
    parser.add_argument('--tensorboard', action='store_true', help="use tensorboard")
    parser.add_argument('--seed', type=int, default=0, help= 'random seed')

    # === CUDA ======================================================================================================= #
    parser.add_argument('--cuda', dest='cuda', action='store_true', help="use CUDA during training")
    parser.add_argument('--no-cuda', dest='cuda', action='store_false', help="do NOT use CUDA during training")
    parser.set_defaults(cuda=True)
    # ================================================================================================================ #

    # Parse given arguments
    args = parser.parse_args()

    # set random seed
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.flow == 'linear':
        args.max_latent_ind = args.num_support_sets
        if args.gan_type == "ProgGAN" or args.gan_type == 'SNGAN_MNIST':
            args.deformator = "PROJECTIVE"
        else:
            args.deformator = "ORTHO"
            args.max_latent_ind = G.dim_z

    # Create output dir and save current arguments
    exp_dir = create_exp_dir(args)

    # CUDA
    use_cuda = False
    multi_gpu = False
    if torch.cuda.is_available():
        if args.cuda:
            use_cuda = True
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
            if torch.cuda.device_count() > 1:
                multi_gpu = True
        else:
            print("*** WARNING ***: It looks like you have a CUDA device, but aren't using CUDA.\n"
                  "                 Run with --cuda for optimal training speed.")
            torch.set_default_tensor_type('torch.FloatTensor')
    else:
        torch.set_default_tensor_type('torch.FloatTensor')

    # Build GAN generator model and load with pre-trained weights
    print("#. Build GAN generator model G and load with pre-trained weights...")
    print("  \\__GAN type: {}".format(args.gan_type))
    if args.gan_type == 'StyleGAN2':
        print("  \\__Search for paths in {}-space".format('W' if args.shift_in_w_space else 'Z'))
    if args.z_truncation:
        print("  \\__Input noise truncation: {}".format(args.z_truncation))
    print("  \\__Pre-trained weights: {}".format(
        GAN_WEIGHTS[args.gan_type]['weights'][args.stylegan2_resolution] if args.gan_type == 'StyleGAN2' else
        GAN_WEIGHTS[args.gan_type]['weights'][GAN_RESOLUTIONS[args.gan_type]]))

    # === BigGAN ===
    if args.gan_type == 'BigGAN':
        G = build_biggan(pretrained_gan_weights=GAN_WEIGHTS[args.gan_type]['weights'][GAN_RESOLUTIONS[args.gan_type]],
                         target_classes=args.biggan_target_classes)
    # === ProgGAN ===
    elif args.gan_type == 'ProgGAN':
        G = build_proggan(pretrained_gan_weights=GAN_WEIGHTS[args.gan_type]['weights'][GAN_RESOLUTIONS[args.gan_type]])
    # === StyleGAN ===
    elif args.gan_type == 'StyleGAN2':
        G = build_stylegan2(pretrained_gan_weights=GAN_WEIGHTS[args.gan_type]['weights'][args.stylegan2_resolution],
                            resolution=args.stylegan2_resolution,
                            shift_in_w_space=args.shift_in_w_space)
    # === Spectrally Normalised GAN (SNGAN) ===
    else:
        G = build_sngan(pretrained_gan_weights=GAN_WEIGHTS[args.gan_type]['weights'][GAN_RESOLUTIONS[args.gan_type]],
                        gan_type=args.gan_type)

    # Build Support Sets model S
    print("#. Build Support Sets S...")
    print("  \\__Number of Support Sets    : {}".format(args.num_support_sets))
    print("  \\__Number of Support Dipoles : {}".format(args.num_support_dipoles))
    print("  \\__Support Vectors dim       : {}".format(G.dim_z))
    print("  \\__Learn RBF alphas          : {}".format(args.learn_alphas))
    print("  \\__Learn RBF gammas          : {}".format(args.learn_gammas))
    if not args.learn_gammas:
        print("  \\__RBF gamma                 : {}".format(1.0 / G.dim_z if args.gamma is None else args.gamma))

    if args.flow == 'FFJORD':
        print('Use FFJORD')
        args.num_support_sets = G.dim_z
        S = cnf(G.dim_z, args.flow_modules, 1)
    elif args.flow == 'linear':
        print('Use LinearGANSpace')
        S = LatentDeformator(shift_dim=G.dim_z,
                             input_dim=args.max_latent_ind,
                             out_dim=G.dim_z,
                             type=DeformatorType[args.deformator]
                             )
    else:
        print('Use RBF')
        S = SupportSets(num_support_sets=args.num_support_sets,
                        num_support_dipoles=args.num_support_dipoles,
                        support_vectors_dim=G.dim_z,
                        learn_alphas=args.learn_alphas,
                        learn_gammas=args.learn_gammas,
                        gamma=1.0 / G.dim_z if args.gamma is None else args.gamma)

    if args.small_dim:
        if args.gan_type == 'ProgGAN' or args.gan_type == 'StyleGAN2':
            args.small_dim = 200
        else:
            args.small_dim = G.dim_z // 2

    # Count number of trainable parameters
    print("  \\__Trainable parameters: {:,}".format(sum(p.numel() for p in S.parameters() if p.requires_grad)))

    # Build reconstructor model R
    print("#. Build reconstructor model R...")

    R = Reconstructor(reconstructor_type=args.reconstructor_type,
                      dim=G.dim_z,
                      channels=1 if args.gan_type == 'SNGAN_MNIST' else 3)

    # Count number of trainable parameters
    print("  \\__Trainable parameters: {:,}".format(sum(p.numel() for p in R.parameters() if p.requires_grad)))

    # Set up trainer
    print("#. Experiment: {}".format(exp_dir))
    trn = Trainer(params=args, exp_dir=exp_dir, use_cuda=use_cuda, multi_gpu=multi_gpu)

    # Train
    trn.train(generator=G, support_sets=S, reconstructor=R)


if __name__ == '__main__':
    main()
