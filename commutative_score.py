import argparse
from audioop import reverse
import cv2
import random
import os
import os.path as osp
from tabnanny import verbose
import numpy as np
import torch
from torch import nn
from PIL import Image
import json
import torchvision
from torchvision import transforms
from torchvision.transforms import ToPILImage
from lib import *
from models.gan_load import build_biggan, build_proggan, build_stylegan2, build_sngan
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Action Units
AUs = {
    "au_1": "Inner_Brow_Raiser",
    "au_2": "Outer_Brow_Raiser",
    "au_4": "Brow_Lowerer",
    "au_5": "Upper_Lid_Raiser",
    "au_6": "Cheek_Raiser",
    "au_9": "Nose_Wrinkler",
    "au_12": "Lip_Corner_Puller",  # (aka "Smile")
    "au_15": "Lip_Corner_Depressor",
    "au_17": "Chin_Raiser",
    "au_20": "Lip_stretcher",
    "au_25": "Lips_part",
    "au_26": "Jaw_Drop"
}


class DataParallelPassthrough(nn.DataParallel):
    def __getattr__(self, name):
        try:
            return super(DataParallelPassthrough, self).__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)


class ModelArgs:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def tensor2image(tensor, img_size=None, adaptive=False):
    # Squeeze tensor image
    tensor = tensor.squeeze(dim=0)
    if adaptive:
        tensor = (tensor - tensor.min()) / (tensor.max() - tensor.min())
        if img_size:
            return ToPILImage()((255 * tensor.cpu().detach()).to(torch.uint8)).resize((img_size, img_size))
        else:
            return ToPILImage()((255 * tensor.cpu().detach()).to(torch.uint8))
    else:
        tensor = (tensor + 1) / 2
        tensor.clamp(0, 1)
        if img_size:
            return ToPILImage()((255 * tensor.cpu().detach()).to(torch.uint8)).resize((img_size, img_size))
        else:
            return ToPILImage()((255 * tensor.cpu().detach()).to(torch.uint8))


def build_gan(gan_type, target_classes, stylegan2_resolution, shift_in_w_space, use_cuda, multi_gpu):
    # -- BigGAN
    if gan_type == 'BigGAN':
        G = build_biggan(pretrained_gan_weights=GAN_WEIGHTS[gan_type]['weights'][GAN_RESOLUTIONS[gan_type]],
                         target_classes=target_classes)
    # -- ProgGAN
    elif gan_type == 'ProgGAN':
        G = build_proggan(pretrained_gan_weights=GAN_WEIGHTS[gan_type]['weights'][GAN_RESOLUTIONS[gan_type]])
    # -- StyleGAN2
    elif gan_type == 'StyleGAN2':
        G = build_stylegan2(pretrained_gan_weights=GAN_WEIGHTS[gan_type]['weights'][stylegan2_resolution],
                            resolution=stylegan2_resolution,
                            shift_in_w_space=shift_in_w_space)
    # -- Spectrally Normalised GAN (SNGAN)
    else:
        G = build_sngan(pretrained_gan_weights=GAN_WEIGHTS[gan_type]['weights'][GAN_RESOLUTIONS[gan_type]],
                        gan_type=gan_type)
    for param in G.G.parameters():
        param.requires_grad = False

    # Upload GAN generator model to GPU
    if use_cuda:
        G = G.cuda()

    # Parallelize GAN generator model into multiple GPUs if possible
    if multi_gpu:
        G = DataParallelPassthrough(G)

    return G


def crop_face(images, idx, bbox, padding=0.0):
    """Crop faces from given images for the given bounding boxes and padding."""
    x_min = int((1.0 - padding) * bbox[0])
    y_min = int((1.0 - padding) * bbox[1])
    x_max = int((1.0 + padding) * bbox[2])
    y_max = int((1.0 + padding) * bbox[3])

    x_min -= 50
    x_max += 50
    y_min -= 50
    y_max += 30
    x_min = max(x_min, 0)
    y_min = max(y_min, 0)

    x_max = min(images.shape[2], x_max)
    y_max = min(images.shape[3], y_max)
    x_min = int(x_min)
    y_min = int(y_min)
    x_max = int(x_max)
    y_max = int(y_max)

    return images[idx, :, x_min:x_max, y_min:y_max].unsqueeze(0)


def manipulation(use_cuda, shift_steps, dim, G, S, args_json, args, w, z, current_path_latent_codes, current_path_latent_shifts, eps):
    cnt = 0
    for _ in range(shift_steps):
        cnt += 1
        # Calculate shift vector based on current z
        support_sets_mask = torch.zeros(1, G.dim_z)
        support_sets_mask[0, dim] = 1.0
        if use_cuda:
            support_sets_mask.cuda()

        # Update z.
        if args_json.__dict__["flow"] == 'FFJORD':
            with torch.no_grad():
                z_new, _ = S(w if args_json.__dict__["shift_in_w_space"] else z,
                            support_sets_mask,
                            torch.tensor(eps),
                            torch.zeros(support_sets_mask.shape[0], 1).to(z.device))
            if args_json.__dict__["shift_in_w_space"]:
                shift = z_new - w
                w = z_new
            else:
                shift = z_new - z
                z = z_new
        elif args_json.__dict__["flow"] == 'linear':
            support_sets_mask = torch.zeros(1, args_json.__dict__['num_support_sets'])
            support_sets_mask[0, dim] = 1.0
            with torch.no_grad():
                shift = S(eps*support_sets_mask)
            if args_json.__dict__["shift_in_w_space"]:
                w = w + shift
            else:
                z = z + shift
        else:
            support_sets_mask = torch.zeros(1, args_json.__dict__['num_support_sets'])
            support_sets_mask[0, dim] = 1.0
            # Get latent space shift vector
            with torch.no_grad():
                shift = eps * S(support_sets_mask, w if args_json.__dict__["shift_in_w_space"] else z)
            # Update z/w
            if args_json.__dict__["shift_in_w_space"]:
                w = w + shift
            else:
                z = z + shift

        # Store latent codes and shifts
        if cnt == args.shift_leap:
            current_path_latent_shifts.append(shift)
            current_path_latent_codes.append(w if args_json.__dict__["shift_in_w_space"] else z)
            cnt = 0
    return w, z, current_path_latent_codes, current_path_latent_shifts

def manipulation_linear(use_cuda, shift_steps, dim, G, S, args_json, args, w, z, current_path_latent_codes, current_path_latent_shifts, eps, signs):
    if args_json.__dict__["shift_in_w_space"]:
        w = w.repeat(sum(shift_steps)+1, 1)
    else:
        z = z.repeat(sum(shift_steps)+1, 1)
    # Calculate shift vector based on current z
    support_sets_mask = torch.zeros(sum(shift_steps), G.dim_z)
    ind = 0
    for i in range(len(shift_steps)):
        for j in range(shift_steps[i]):
            support_sets_mask[ind, dim[i]] = eps[i] if signs[i] == '+' else -eps[i]
            ind += 1
    if use_cuda:
        support_sets_mask.cuda()

    # Update z.
    if args_json.__dict__["flow"] == 'FFJORD':
        with torch.no_grad():
            z_new = S.manipulate(w if args_json.__dict__["shift_in_w_space"] else z,
                           support_sets_mask)
        if args_json.__dict__["shift_in_w_space"]:
            shift = z_new - w
            w = z_new
        else:
            shift = z_new - z
            z = z_new
    else:
        support_sets_mask = torch.zeros(sum(shift_steps)+1, args_json.__dict__['num_support_sets'])
        ind = 0
        for i in range(len(shift_steps)):
            for j in range(shift_steps[i]):
                support_sets_mask[ind, dim[i]] = eps[i] if signs[i] == '+' else -eps[i]
                ind += 1
        if use_cuda:
            support_sets_mask.cuda()
        with torch.no_grad():
            shift = S(support_sets_mask)
        if args_json.__dict__["shift_in_w_space"]:
            for i in range(sum(shift_steps)):
                w[i+1] = w[i] + shift[i]
            shift = torch.cat(((w[0]-w[0]).unsqueeze(0), shift))
        else:
            for i in range(sum(shift_steps)):
                z[i+1] = z[i] + shift[i]
            shift = torch.cat(((z[0]-z[0]).unsqueeze(0), shift))
    
    latent_shape = w.shape[0] if args_json.__dict__["shift_in_w_space"] else z.shape[0]

    for i in range(latent_shape-1):
        current_path_latent_shifts.append(shift[i+1].unsqueeze(0))
        current_path_latent_codes.append(w[i+1].unsqueeze(0) if args_json.__dict__["shift_in_w_space"] else z[i+1].unsqueeze(0))
    return current_path_latent_codes, current_path_latent_shifts

def get_args():
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('-v', '--verbose', action='store_true', help="set verbose mode on")
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--exp', type=str, required=True, help="set experiment's model dir (created by `train.py` and "
                                                               "used by `traverse_latent_space.py`.)")
    parser.add_argument('--pool', type=str, required=True, help="choose pool of pre-defined latent codes and their "
                                                                "latent traversals (should be a subdirectory of "
                                                                "experiments/<exp>/results/, as created by "
                                                                "traverse_latent_space.py)")
    parser.add_argument('--shift-steps', type=str, default='30-30', help="number of shifts per positive/negative path "
                                                                    "direction")
    parser.add_argument('--attr', type=str, default='age-smile', help='attributes')
    parser.add_argument('--signs', type=str, default='++', help='attributes signs')
    parser.add_argument('--eps', type=str, default='1-1' ,help="shift magnitude")
    parser.add_argument('--shift-leap', type=int, default=1,
                        help="set path shift leap (after how many steps to generate images)")
    parser.add_argument('--batch-size', type=int, help="set generator batch size (if not set, use the total number of "
                                                       "images per path)")
    parser.add_argument('--img-size', type=int, help="set size of saved generated images (if not set, use the output "
                                                     "size of the respective GAN generator)")
    parser.add_argument('--img-quality', type=int, default=95, help="set JPEG image quality")
    parser.add_argument('--gif', action='store_true', help="Create GIF traversals")
    parser.add_argument('--gif-size', type=int, default=256, help="set gif resolution")
    parser.add_argument('--gif-fps', type=int, default=30, help="set gif frame rate")
    parser.add_argument('--dim', type=str, default='default', help="set translate dimensions (xx-xx)")
    # ================================================================================================================ #
    parser.add_argument('--cuda', dest='cuda', action='store_true', help="use CUDA during training")
    parser.add_argument('--no-cuda', dest='cuda', action='store_false', help="do NOT use CUDA during training")
    parser.set_defaults(cuda=True)
    parser.set_defaults(verbose=True)
    return parser.parse_args()

def main():
    args = get_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # Check structure of `args.exp`
    if not osp.isdir(args.exp):
        raise NotADirectoryError("Invalid given directory: {}".format(args.exp))

    # -- args.json file (pre-trained model arguments)
    args_json_file = osp.join(args.exp, 'args.json')
    if not osp.isfile(args_json_file):
        raise FileNotFoundError("File not found: {}".format(args_json_file))
    args_json = ModelArgs(**json.load(open(args_json_file)))
    gan_type = args_json.__dict__["gan_type"]

    # -- models directory (support sets and reconstructor, final or checkpoint files)
    models_dir = osp.join(args.exp, 'models')
    if not osp.isdir(models_dir):
        raise NotADirectoryError("Invalid models directory: {}".format(models_dir))

    # ---- Get all files of models directory
    models_dir_files = [f for f in os.listdir(models_dir) if osp.isfile(osp.join(models_dir, f))]

    # ---- Check for support sets file (final or checkpoint)
    support_sets_model = osp.join(models_dir, 'support_sets.pt')
    if not osp.isfile(support_sets_model):
        support_sets_checkpoint_files = []
        for f in models_dir_files:
            if 'support_sets-' in f:
                support_sets_checkpoint_files.append(f)
        support_sets_checkpoint_files.sort()
        support_sets_model = osp.join(models_dir, support_sets_checkpoint_files[-1])

    # Check given pool directory
    pool = osp.join('experiments', 'latent_codes')
    if gan_type == 'BigGAN':
        biggan_target_classes = ''
        for c in args_json.__dict__["biggan_target_classes"]:
            biggan_target_classes += '-{}'.format(c)
        pool = osp.join(pool, gan_type + biggan_target_classes, args.pool)
    else:
        pool = osp.join(pool, gan_type, args.pool)

    if not osp.isdir(pool):
        raise NotADirectoryError("Invalid pool directory: {} -- Please run sample_gan.py to create it.".format(pool))

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
    if args.verbose:
        print("#. Build GAN generator model G and load with pre-trained weights...")
        print("  \\__GAN type: {}".format(gan_type))
        print("  \\__Pre-trained weights: {}".format(
            GAN_WEIGHTS[gan_type]['weights'][args_json.__dict__["stylegan2_resolution"]]
            if gan_type == 'StyleGAN2' else GAN_WEIGHTS[gan_type]['weights'][GAN_RESOLUTIONS[gan_type]]))

    G = build_gan(gan_type=gan_type,
                  target_classes=args_json.__dict__["biggan_target_classes"],
                  stylegan2_resolution=args_json.__dict__["stylegan2_resolution"],
                  shift_in_w_space=args_json.__dict__["shift_in_w_space"],
                  use_cuda=use_cuda,
                  multi_gpu=multi_gpu).eval()

    # Define SFD face detector model
    face_detector = SFDDetector(path_to_detector='models/pretrained/sfd/s3fd-619a316812.pth',
                                device="cuda" if use_cuda else "cpu")
    face_detector_trans = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(256)])

    # Define ID comparator based on ArcFace
    id_comp = IDComparator()
    id_comp.eval()
    if use_cuda:
        id_comp.cuda()

    # Define FairFace model for predicting gender, age, and race
    fairface = torchvision.models.resnet34(pretrained=True)
    fairface.fc = nn.Linear(fairface.fc.in_features, 18)
    fairface.load_state_dict(torch.load('models/pretrained/fairface/fairface_alldata_4race_20191111.pt'))
    fairface.eval()
    if use_cuda:
        fairface.cuda()

    # Define Hopenet pose estimator
    hopenet = Hopenet(torchvision.models.resnet.Bottleneck, [3, 4, 6, 3], 66)
    hopenet.load_state_dict(torch.load('models/pretrained/hopenet/hopenet_alpha2.pkl'))
    hopenet.eval()
    if use_cuda:
        hopenet.cuda()

    hopenet_softmax = nn.Softmax(dim=1)
    if use_cuda:
        hopenet_softmax.cuda()

    idx_tensor = [idx for idx in range(66)]
    idx_tensor = torch.FloatTensor(idx_tensor)
    if use_cuda:
        idx_tensor = idx_tensor.cuda()

    # Define face transformation required by Hopenet and FairFace
    face_trans_hope_fair = transforms.Compose([transforms.Resize(224),
                                               transforms.CenterCrop(224),
                                               transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                    std=[0.229, 0.224, 0.225])])

    # Define CelebA attributes predictor
    celeba_5 = celeba_attr_predictor(attr_file='lib/evaluation/celeba_attributes/attributes_5.json',
                                     pretrained='models/pretrained/celeba_attributes/eval_predictor.pth.tar').eval()
    celeba_5_trans = transforms.Compose([transforms.Resize(224),
                                         transforms.CenterCrop(224),
                                         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    celeba_5_softmax = nn.Softmax(dim=1)

    if use_cuda:
        celeba_5.cuda()

    # Build support sets model S
    if args.verbose:
        print("#. Build support sets model S...")

    if args_json.__dict__["flow"] == 'FFJORD':
        print('Use FFJORD')
        S = cnf(G.dim_z, args_json.__dict__["flow_modules"], 1)
    elif args_json.__dict__["flow"] == 'linear':
        print('Use LinearGANSpace')
        S = LatentDeformator(shift_dim=G.dim_z,
                             input_dim=args_json.__dict__['num_support_sets'],
                             out_dim=G.dim_z,
                             type=DeformatorType[args_json.__dict__['deformator']])
    else:
        print('Use RBF')
        S = SupportSets(num_support_sets=args_json.__dict__["num_support_sets"],
                        num_support_dipoles=args_json.__dict__["num_support_dipoles"],
                        support_vectors_dim=G.dim_z,
                        learn_alphas=args_json.__dict__["learn_alphas"],
                        learn_gammas=args_json.__dict__["learn_gammas"],
                        gamma=1.0 / G.dim_z if args_json.__dict__["gamma"] is None else args_json.__dict__["gamma"])

    # Load pre-trained weights and set to evaluation mode
    if args.verbose:
        print("  \\__Pre-trained weights: {}".format(support_sets_model))
    S.load_state_dict(torch.load(support_sets_model, map_location=lambda storage, loc: storage))
    for param in S.parameters():
        param.requires_grad = False
    if args.verbose:
        print("  \\__Set to evaluation mode")
    S.eval()

    # Upload support sets model to GPU
    if use_cuda:
        S = S.cuda()
    
    shift_steps = args.shift_steps.split('-')
    shift_steps = [int(s) for s in shift_steps]

    attr_list =  args.attr.split('-')
    if args_json.__dict__["flow"] == 'FFJORD' or args_json.__dict__["flow"] == 'linear':
        eps = args.eps.split('-')
        eps = [float(e) for e in eps]
        eps_reverse = list(reversed(eps))
    else:
        args.eps = float(args.eps)
    if args.dim == 'default':
        dim = [0, 1]
    else:
        dim = args.dim.split('-')
        dim = [int(d) for d in dim]
    # signs = '-+'  # '--'
    args.signs = args.signs.split('-')
    signs = ''
    for sign in args.signs:
        signs = signs + '+' if sign == 'plus' else signs + '-'
    dim_reverse = list(reversed(dim))
    signs_reverse = signs[::-1]
    shift_steps_reverse = list(reversed(shift_steps))
    attr_list_reverse = list(reversed(attr_list))

    # Create output dir for generated images
    out_dir = osp.join(args.exp, 'results', args.pool,
                       '{}_{}_{}'.format(args.shift_steps, args.eps, round(sum(shift_steps), 3)))
    os.makedirs(out_dir, exist_ok=True)

    # Set default batch size
    if args.batch_size is None:
        args.batch_size = 1

    ## ============================================================================================================== ##
    ##                                                                                                                ##
    ##                                              [Latent Codes Pool]                                               ##
    ##                                                                                                                ##
    ## ============================================================================================================== ##
    # Get latent codes from the given pool
    if args.verbose:
        print("#. Use latent codes from pool {}...".format(args.pool))
    latent_codes_dirs = [dI for dI in os.listdir(pool) if os.path.isdir(os.path.join(pool, dI))]
    latent_codes_dirs.sort()
    latent_codes = []
    for subdir in latent_codes_dirs:
        latent_codes.append(torch.load(osp.join(pool, subdir, 'latent_code.pt'),
                                       map_location=lambda storage, loc: storage))
    zs = torch.cat(latent_codes)
    num_of_latent_codes = zs.size()[0]

    if use_cuda:
        zs = zs.cuda()

    ## ============================================================================================================== ##
    ##                                                                                                                ##
    ##                                            [Latent space traversal]                                            ##
    ##                                                                                                                ##
    ## ============================================================================================================== ##
    if args.verbose:
        print("#. Traverse latent space...")
        print("  \\__Experiment       : {}".format(osp.basename(osp.abspath(args.exp))))
        print("  \\__Shift magnitude  : {}".format(args.eps))
        print("  \\__Shift steps      : {}".format(args.shift_steps))
        print("  \\__Traversal length : {}".format(round(sum(shift_steps), 3)))
        print("  \\__Save results at  : {}".format(out_dir))

    if 'age' in attr_list:
        faces_age_diffs_pre = []
        age_original_diffs = []
    if 'gender' in attr_list:
        gender_original_diffs = []
        faces_gender_diffs_pre = []
    if 'race' in attr_list:
        race_original_diffs = []
        faces_race_diffs_pre = []
    if 'smile' in attr_list:
        smile_original_diffs = []
        faces_smile_diffs_pre = []
    if 'bangs' in attr_list:
        bangs_original_diffs = []
        faces_bangs_diffs_pre = []
    if 'beard' in attr_list:
        beard_original_diffs = []
        faces_beard_diffs_pre = []
    if 'roll' in attr_list:
        roll_original_diffs = []
        faces_roll_diffs_pre = []
    if 'pitch' in attr_list:
        pitch_original_diffs = []
        faces_pitch_diffs_pre = []
    if 'yaw' in attr_list:
        yaw_original_diffs = []
        faces_yaw_diffs_pre = []

    # Iterate over given latent codes(num og original images)
    for i in range(num_of_latent_codes):
        # Un-squeeze current latent code in shape [1, dim] and create hash code for it
        z_ = zs[i, :].unsqueeze(0)

        latent_code_hash = latent_codes_dirs[i]
        if args.verbose:
            update_progress("  \\__.Latent code hash: {} [{:03d}/{:03d}] ".format(latent_code_hash,
                                                                                  i+1,
                                                                                  num_of_latent_codes),
                            num_of_latent_codes, i)

        # Create directory for current latent code
        latent_code_dir = osp.join(out_dir, '{}'.format(latent_code_hash))
        os.makedirs(latent_code_dir, exist_ok=True)

        # Create directory for storing path images
        transformed_images_root_dir = osp.join(latent_code_dir, 'paths_images_eval_attr')
        os.makedirs(transformed_images_root_dir, exist_ok=True)

        ## ========================================================================================================== ##
        ##                                                                                                            ##
        ##                                             [ Path Traversal ]                                             ##
        ##                                                                                                            ##
        ## ========================================================================================================== ##
        # Iterate over (interpretable) directions
        # Create shifted latent codes (for the given latent code z) and generate transformed images
        transformed_images = []

        # Current path's latent codes and shifts lists
        current_path_latent_codes = [G.get_w(z_) if args_json.__dict__["shift_in_w_space"] else z_]
        current_path_latent_shifts = [torch.zeros_like(z_).cuda() if use_cuda else torch.zeros_like(z_)]

        ## ====================================================================================================== ##
        ##                                                                                                        ##
        ##                    [ Traverse through current path (positive/negative directions) ]                    ##
        ##                                                                                                        ##
        ## ====================================================================================================== ##
        if args_json.__dict__["shift_in_w_space"]:
            z = z_.clone()
            w = G.get_w(z)
        else:
            z = z_.clone()
            w = None
        transformed_images_dir = osp.join(transformed_images_root_dir, 'path_{}'.format(dim[0]))
        for d in range(1, len(dim)):
            transformed_images_dir = transformed_images_dir + 'to{}'.format(dim[d])
        img_path = [osp.join(transformed_images_dir, '{:06d}.jpg'.format(0))]
        for j in range(len(shift_steps)):
            img_path.append(osp.join(transformed_images_dir, '{:06d}.jpg'.format(sum(shift_steps[0: j+1]))))

        if not os.path.isfile(img_path[-1]):
            if args_json.__dict__["flow"] == 'FFJORD' or args_json.__dict__["flow"] == 'linear':
                current_path_latent_codes, current_path_latent_shifts = manipulation_linear(use_cuda, shift_steps, dim, G, S, args_json, args, w, z, current_path_latent_codes, current_path_latent_shifts, eps, signs)
            else:
                for j in range(len(attr_list)):
                    eps = args.eps if signs[j] == '+' else -args.eps
                    w, z, current_path_latent_codes, current_path_latent_shifts = manipulation(use_cuda, shift_steps[j], dim[j], G, S, args_json, args, w, z, current_path_latent_codes, current_path_latent_shifts, eps)
            # Generate transformed images
            # Split latent codes and shifts in batches
            current_path_latent_codes = torch.cat(current_path_latent_codes)
            current_path_latent_codes_batches = torch.split(current_path_latent_codes, args.batch_size)
            current_path_latent_shifts = torch.cat(current_path_latent_shifts)
            current_path_latent_shifts_batches = torch.split(current_path_latent_shifts, args.batch_size)

            if len(current_path_latent_codes_batches) != len(current_path_latent_shifts_batches):
                raise AssertionError()
            else:
                num_batches = len(current_path_latent_codes_batches)

            transformed_img = []
            for t in range(num_batches):
                with torch.no_grad():
                    if args_json.__dict__["shift_in_w_space"]:
                        transformed_img.append(G(z=current_path_latent_codes_batches[t],
                                                    latent_is_w=True))
                    else:
                        transformed_img.append(G(z=current_path_latent_codes_batches[t]))
            transformed_img = torch.cat(transformed_img)

            # Convert tensors (transformed images) into PIL images
            for t in range(transformed_img.size()[0]):
                transformed_images.append(tensor2image(transformed_img[t, :].cpu(),
                                                        img_size=args.img_size,
                                                        adaptive=True))
            # Save all images in `transformed_images` list under `transformed_images_root_dir/<path_<dim>/`
            os.makedirs(transformed_images_dir, exist_ok=True)
            for t in range(len(transformed_images)):
                transformed_images[t].save(osp.join(transformed_images_dir, '{:06d}.jpg'.format(t)),
                                            "JPEG", quality=args.img_quality, optimize=True, progressive=True)
        #######################################################################
        # calculate scores using pretrained attribute predictors
        #######################################################################
        imgs = []
        for path in img_path:
            img = cv2.imread(path, cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype('uint8')
            imgs.append(img)
        imgs = np.array(imgs)
        path_images_tensor = torch.tensor(np.transpose(imgs, (0, 3, 1, 2))).float()
        if use_cuda:
            path_images_tensor = path_images_tensor.to('cuda:0')
        ########################################################################################################
        ##                                                                                                    ##
        ##                                        [ Face Detection ]                                          ##
        ##                                                                                                    ##
        ########################################################################################################
        # Detect faces in path images (B x 3 x 256 x 256
        with torch.no_grad():
            detected_faces, _, _ = face_detector.detect_from_batch(face_detector_trans(path_images_tensor))
        ########################################################################################################

        ########################################################################################################
        ##                                                                                                    ##
        ##                                     [ Face Bounding Boxes ]                                        ##
        ##                                                                                                    ##
        ########################################################################################################
        face_bbox_list = []
        face_w = []
        face_h = []
        for t in range(len(detected_faces)):
            if len(detected_faces[t]) > 0:
                face_bbox = detected_faces[t][0].tolist()
                face_bbox_list.append(face_bbox)
                face_w.append((face_bbox[2] - face_bbox[0]) / 256.0)
                face_h.append((face_bbox[3] - face_bbox[1]) / 256.0)
            else:
                face_w.append(256.0)
                face_h.append(256.0)
        ########################################################################################################

        ########################################################################################################
        ##                                                                                                    ##
        ##                                      [ CelebA Attributes ]                                         ##
        ##                                                                                                    ##
        ########################################################################################################
        # The following are based on [7] (see above):                                                         ##
        # -- Bangs      : Proportion of the exposed forehead (100%, 80%, 60%, 40%, 20%, and 0%).              ##
        # -- Eyeglasses : Thickness of glasses frames and type of glasses (ordinary / sunglasses).            ##
        # -- Beard      : Thickness of the beard.                                                             ##
        # -- Smiling    : Ratio of exposed teeth and open mouth.                                              ##
        # -- Age        : below 15, 15-30, 30-40, 40-50, 50-60, and above 60.                                 ##
        ########################################################################################################
        if gan_type == 'StyleGAN2':
            # According to [7], transform image to range [-1, 1] before mean-std normalization
            with torch.no_grad():
                attribute_predictions = celeba_5(
                    celeba_5_trans(path_images_tensor.div(255.0).mul(2.0).add(-1.0)))
        else:
            # According to [7], transform image to range [0, 1] before mean-std normalization
            path_images_tensor_ = path_images_tensor.clone()
            path_images_tensor_ = (path_images_tensor_ - path_images_tensor_.min()) / \
                                    (path_images_tensor_.max() - path_images_tensor_.min())
            with torch.no_grad():
                attribute_predictions = celeba_5(celeba_5_trans(path_images_tensor_))
        celeba_attr = {}
        for attr, attr_predictions in attribute_predictions.items():
            attr_scores = celeba_5_softmax(attr_predictions).cpu().numpy()
            scores = np.max(attr_scores, axis=1)
            labels = np.argmax(attr_scores, axis=1)
            final_scores = (labels + scores) / 6.0
            celeba_attr.update({attr: final_scores})

        ########################################################################################################
        ##                                                                                                    ##
        ##                                    [ Gender / Age  / Race ]                                        ##
        ##                                                                                                    ##
        ########################################################################################################
        cropped_faces = torch.zeros(len(detected_faces), 3, 224, 224)
        for t in range(len(detected_faces)):
            cropped_faces[t] = face_trans_hope_fair(crop_face(images=face_detector_trans(path_images_tensor),
                                                                idx=t,
                                                                bbox=detected_faces[t][0][:-1]
                                                                if len(detected_faces[t]) > 0
                                                                else [0, 0, 256, 256],
                                                                padding=0.25).div(255.0))
        if use_cuda:
            cropped_faces = cropped_faces.cuda()

        with torch.no_grad():
            outputs = fairface(cropped_faces).cpu().detach().numpy()

        # Gender predictions
        gender_outputs = outputs[:, 7:9]
        gender_scores = np.exp(gender_outputs) / np.sum(np.exp(gender_outputs), axis=1, keepdims=True)
        femaleness_scores = gender_scores[:, 1].tolist()

        # Age predictions
        age_outputs = outputs[:, 9:18]
        age_scores = np.exp(age_outputs) / np.sum(np.exp(age_outputs), axis=1, keepdims=True)
        age_predictions = np.argmax(age_scores, axis=1)
        # Assign continuous scores
        age_predictions = (age_predictions + np.max(age_scores, axis=1)) / 9.0

        # Race Prediction
        # 0: 'white'
        # 1: 'East Asian'
        # 2: 'Latino_Hispanic'
        # 3: 'Southeast Asian'
        # 4: 'Indian'
        # 5: 'Middle Eastern'
        # 6: 'Black'
        race_outputs = outputs[:, :7]
        race_scores = np.exp(race_outputs) / np.sum(np.exp(race_outputs), axis=1, keepdims=True)
        race_predictions = np.argmax(race_scores, axis=1)
        # Assign continuous scores
        race_predictions = (race_predictions + np.max(race_scores, axis=1)) / 7.0
        ########################################################################################################
        ##                                                                                                    ##
        ##                                              [ Pose ]                                              ##
        ##                                                                                                    ##
        ########################################################################################################
        cropped_faces = torch.zeros(len(detected_faces), 3, 224, 224)
        for t in range(len(detected_faces)):
            cropped_faces[t] = face_trans_hope_fair(crop_face(images=face_detector_trans(path_images_tensor),
                                                                idx=t,
                                                                bbox=detected_faces[t][0][:-1]
                                                                if len(detected_faces[t]) > 0
                                                                else [0, 0, 256, 256]).div(255.0))
        if use_cuda:
            cropped_faces = cropped_faces.cuda()

        # Predict yaw, pitch, roll in degrees
        with torch.no_grad():
            yaw, pitch, roll = hopenet(cropped_faces)
        yaw_predicted = hopenet_softmax(yaw)
        pitch_predicted = hopenet_softmax(pitch)
        roll_predicted = hopenet_softmax(roll)
        yaw_predicted = torch.sum(yaw_predicted * idx_tensor, 1) * 3 - 99
        pitch_predicted = torch.sum(pitch_predicted * idx_tensor, 1) * 3 - 99
        roll_predicted = torch.sum(roll_predicted * idx_tensor, 1) * 3 - 99
        ########################################################################################################
        if 'age' in attr_list:
            age_forward = age_predictions[-1]
            age_index = attr_list.index('age')
            age_original_diffs.append(np.abs(age_predictions[age_index+1] - age_predictions[age_index]))
        if 'gender' in attr_list:
            gender_forward = femaleness_scores[-1]
            gender_index = attr_list.index('gender')
            gender_original_diffs.append(np.abs(femaleness_scores[gender_index+1] - femaleness_scores[gender_index]))
        if 'race' in attr_list:
            race_forward = race_predictions[-1]
            race_index = attr_list.index('race')
            race_original_diffs.append(np.abs(race_predictions[race_index+1] - race_predictions[race_index]))
        if 'smile' in attr_list:
            smile_forward = celeba_attr['Smiling'][-1]
            smile_index = attr_list.index('smile')
            smile_original_diffs.append(np.abs(celeba_attr['Smiling'][smile_index+1] - celeba_attr['Smiling'][smile_index]))
        if 'bangs' in attr_list:
            bangs_forward = celeba_attr['Bangs'][-1]
            bangs_index = attr_list.index('bangs')
            bangs_original_diffs.append(np.abs(celeba_attr['Bangs'][bangs_index+1] - celeba_attr['Bangs'][bangs_index]))
        if 'beard' in attr_list:
            beard_forward = celeba_attr['No_Beard'][-1]
            beard_index = attr_list.index('beard')
            beard_original_diffs.append(np.abs(celeba_attr['No_Beard'][beard_index+1] - celeba_attr['No_Beard'][beard_index]))
        if 'yaw' in attr_list:
            yaw_forward = yaw_predicted.cpu().numpy()[-1]
            yaw_index = attr_list.index('yaw')
            yaw_original_diffs.append(np.abs(yaw_predicted.cpu().numpy()[yaw_index+1] - yaw_predicted.cpu().numpy()[yaw_index]))
        if 'roll' in attr_list:
            roll_forward = roll_predicted.cpu().numpy()[-1]
            roll_index = attr_list.index('roll')
            roll_original_diffs.append(np.abs(roll_predicted.cpu().numpy()[roll_index+1] - roll_predicted.cpu().numpy()[roll_index]))
        if 'pitch' in attr_list:
            pitch_forward = pitch_predicted.cpu().numpy()[-1]
            pitch_index = attr_list.index('pitch')
            pitch_original_diffs.append(np.abs(pitch_predicted.cpu().numpy()[pitch_index+1] - pitch_predicted.cpu().numpy()[pitch_index]))

        
        ## ========================================================================================================== ##
        ##                                                                                                            ##
        ##                                             [ Path Traversal ]                                             ##
        ##                                                                                                            ##
        ## ========================================================================================================== ##
        # Iterate over (interpretable) directions
        # Create shifted latent codes (for the given latent code z) and generate transformed images
        transformed_images = []

        # Current path's latent codes and shifts lists
        current_path_latent_codes = [G.get_w(z_) if args_json.__dict__["shift_in_w_space"] else z_]
        current_path_latent_shifts = [torch.zeros_like(z_).cuda() if use_cuda else torch.zeros_like(z_)]

        ## ====================================================================================================== ##
        ##                                                                                                        ##
        ##                    [ Traverse through current path (positive/negative directions) ]                    ##
        ##                                                                                                        ##
        ## ====================================================================================================== ##
        if args_json.__dict__["shift_in_w_space"]:
            z = z_.clone()
            w = G.get_w(z)
        else:
            z = z_.clone()
            w = None
        transformed_images_dir = osp.join(transformed_images_root_dir, 'path_{}'.format(dim_reverse[0]))
        for d in range(1, len(dim_reverse)):
            transformed_images_dir = transformed_images_dir + 'to{}'.format(dim_reverse[d])
        img_path = [osp.join(transformed_images_dir, '{:06d}.jpg'.format(0))]
        for j in range(len(shift_steps_reverse)):
            img_path.append(osp.join(transformed_images_dir, '{:06d}.jpg'.format(sum(shift_steps_reverse[0: j+1]))))
        if not os.path.isfile(img_path[-1]):
            if args_json.__dict__["flow"] == 'FFJORD' or args_json.__dict__["flow"] == 'linear':
                current_path_latent_codes, current_path_latent_shifts = manipulation_linear(use_cuda, shift_steps_reverse, dim_reverse, G, S, args_json, args, w, z, current_path_latent_codes, current_path_latent_shifts, eps_reverse, signs_reverse)
            else:
                for j in range(len(attr_list)):
                    eps = args.eps if signs_reverse[j] == '+' else -args.eps
                    w, z, current_path_latent_codes, current_path_latent_shifts = manipulation(use_cuda, shift_steps_reverse[j], dim_reverse[j], G, S, args_json, args, w, z, current_path_latent_codes, current_path_latent_shifts, eps)
            # Generate transformed images
            # Split latent codes and shifts in batches
            current_path_latent_codes = torch.cat(current_path_latent_codes)
            current_path_latent_codes_batches = torch.split(current_path_latent_codes, args.batch_size)
            current_path_latent_shifts = torch.cat(current_path_latent_shifts)
            current_path_latent_shifts_batches = torch.split(current_path_latent_shifts, args.batch_size)

            if len(current_path_latent_codes_batches) != len(current_path_latent_shifts_batches):
                raise AssertionError()
            else:
                num_batches = len(current_path_latent_codes_batches)

            transformed_img = []
            for t in range(num_batches):
                with torch.no_grad():
                    if args_json.__dict__["shift_in_w_space"]:
                        transformed_img.append(G(z=current_path_latent_codes_batches[t],
                                                    latent_is_w=True))
                    else:
                        transformed_img.append(G(z=current_path_latent_codes_batches[t]))
            transformed_img = torch.cat(transformed_img)

            # Convert tensors (transformed images) into PIL images
            for t in range(transformed_img.size()[0]):
                transformed_images.append(tensor2image(transformed_img[t, :].cpu(),
                                                        img_size=args.img_size,
                                                        adaptive=True))
            # Save all images in `transformed_images` list under `transformed_images_root_dir/<path_<dim>/`
            os.makedirs(transformed_images_dir, exist_ok=True)
            for t in range(len(transformed_images)):
                transformed_images[t].save(osp.join(transformed_images_dir, '{:06d}.jpg'.format(t)),
                                            "JPEG", quality=args.img_quality, optimize=True, progressive=True)
        #######################################################################
        # caluculate scores using pretrained networks
        #######################################################################
        imgs = []
        for path in img_path:
            img = cv2.imread(path, cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype('uint8')
            imgs.append(img)
        imgs = np.array(imgs)
        path_images_tensor = torch.tensor(np.transpose(imgs, (0, 3, 1, 2))).float()
        if use_cuda:
            path_images_tensor = path_images_tensor.to('cuda:0')
        ########################################################################################################
        ##                                                                                                    ##
        ##                                        [ Face Detection ]                                          ##
        ##                                                                                                    ##
        ########################################################################################################
        # Detect faces in path images (B x 3 x 256 x 256
        with torch.no_grad():
            detected_faces, _, _ = face_detector.detect_from_batch(face_detector_trans(path_images_tensor))
        ########################################################################################################

        ########################################################################################################
        ##                                                                                                    ##
        ##                                     [ Face Bounding Boxes ]                                        ##
        ##                                                                                                    ##
        ########################################################################################################
        face_bbox_list = []
        face_w = []
        face_h = []
        for t in range(len(detected_faces)):
            if len(detected_faces[t]) > 0:
                face_bbox = detected_faces[t][0].tolist()
                face_bbox_list.append(face_bbox)
                face_w.append((face_bbox[2] - face_bbox[0]) / 256.0)
                face_h.append((face_bbox[3] - face_bbox[1]) / 256.0)
            else:
                face_w.append(256.0)
                face_h.append(256.0)
        ########################################################################################################

        ########################################################################################################
        ##                                                                                                    ##
        ##                                      [ CelebA Attributes ]                                         ##
        ##                                                                                                    ##
        ########################################################################################################
        # The following are based on [7] (see above):                                                         ##
        # -- Bangs      : Proportion of the exposed forehead (100%, 80%, 60%, 40%, 20%, and 0%).              ##
        # -- Eyeglasses : Thickness of glasses frames and type of glasses (ordinary / sunglasses).            ##
        # -- Beard      : Thickness of the beard.                                                             ##
        # -- Smiling    : Ratio of exposed teeth and open mouth.                                              ##
        # -- Age        : below 15, 15-30, 30-40, 40-50, 50-60, and above 60.                                 ##
        ########################################################################################################
        if gan_type == 'StyleGAN2':
            # According to [7], transform image to range [-1, 1] before mean-std normalization
            with torch.no_grad():
                attribute_predictions = celeba_5(
                    celeba_5_trans(path_images_tensor.div(255.0).mul(2.0).add(-1.0)))
        else:
            # According to [7], transform image to range [0, 1] before mean-std normalization
            path_images_tensor_ = path_images_tensor.clone()
            path_images_tensor_ = (path_images_tensor_ - path_images_tensor_.min()) / \
                                    (path_images_tensor_.max() - path_images_tensor_.min())
            with torch.no_grad():
                attribute_predictions = celeba_5(celeba_5_trans(path_images_tensor_))
        celeba_attr = {}
        for attr, attr_predictions in attribute_predictions.items():
            attr_scores = celeba_5_softmax(attr_predictions).cpu().numpy()
            scores = np.max(attr_scores, axis=1)
            labels = np.argmax(attr_scores, axis=1)
            final_scores = (labels + scores) / 6.0
            celeba_attr.update({attr: final_scores})

        ########################################################################################################
        ##                                                                                                    ##
        ##                                    [ Gender / Age  / Race ]                                        ##
        ##                                                                                                    ##
        ########################################################################################################
        cropped_faces = torch.zeros(len(detected_faces), 3, 224, 224)
        for t in range(len(detected_faces)):
            cropped_faces[t] = face_trans_hope_fair(crop_face(images=face_detector_trans(path_images_tensor),
                                                                idx=t,
                                                                bbox=detected_faces[t][0][:-1]
                                                                if len(detected_faces[t]) > 0
                                                                else [0, 0, 256, 256],
                                                                padding=0.25).div(255.0))
        if use_cuda:
            cropped_faces = cropped_faces.cuda()

        with torch.no_grad():
            outputs = fairface(cropped_faces).cpu().detach().numpy()

        # Gender predictions
        gender_outputs = outputs[:, 7:9]
        gender_scores = np.exp(gender_outputs) / np.sum(np.exp(gender_outputs), axis=1, keepdims=True)
        femaleness_scores = gender_scores[:, 1].tolist()

        # Age predictions
        age_outputs = outputs[:, 9:18]
        age_scores = np.exp(age_outputs) / np.sum(np.exp(age_outputs), axis=1, keepdims=True)
        age_predictions = np.argmax(age_scores, axis=1)
        # Assign continuous scores
        age_predictions = (age_predictions + np.max(age_scores, axis=1)) / 9.0

        # Race Prediction
        # 0: 'white'
        # 1: 'East Asian'
        # 2: 'Latino_Hispanic'
        # 3: 'Southeast Asian'
        # 4: 'Indian'
        # 5: 'Middle Eastern'
        # 6: 'Black'
        race_outputs = outputs[:, :7]
        race_scores = np.exp(race_outputs) / np.sum(np.exp(race_outputs), axis=1, keepdims=True)
        race_predictions = np.argmax(race_scores, axis=1)
        # Assign continuous scores
        race_predictions = (race_predictions + np.max(race_scores, axis=1)) / 7.0
        ########################################################################################################
        ##                                                                                                    ##
        ##                                              [ Pose ]                                              ##
        ##                                                                                                    ##
        ########################################################################################################
        cropped_faces = torch.zeros(len(detected_faces), 3, 224, 224)
        for t in range(len(detected_faces)):
            cropped_faces[t] = face_trans_hope_fair(crop_face(images=face_detector_trans(path_images_tensor),
                                                                idx=t,
                                                                bbox=detected_faces[t][0][:-1]
                                                                if len(detected_faces[t]) > 0
                                                                else [0, 0, 256, 256]).div(255.0))
        if use_cuda:
            cropped_faces = cropped_faces.cuda()

        # Predict yaw, pitch, roll in degrees
        with torch.no_grad():
            yaw, pitch, roll = hopenet(cropped_faces)
        yaw_predicted = hopenet_softmax(yaw)
        pitch_predicted = hopenet_softmax(pitch)
        roll_predicted = hopenet_softmax(roll)
        yaw_predicted = torch.sum(yaw_predicted * idx_tensor, 1) * 3 - 99
        pitch_predicted = torch.sum(pitch_predicted * idx_tensor, 1) * 3 - 99
        roll_predicted = torch.sum(roll_predicted * idx_tensor, 1) * 3 - 99

        ########################################################################################################
        if 'age' in attr_list:
            faces_age_diffs_pre.append(np.abs(age_predictions[-1] - age_forward))
            age_index = attr_list_reverse.index('age')
            age_original_diffs.append(np.abs(age_predictions[age_index+1] - age_predictions[age_index]))
        if 'gender' in attr_list:
            faces_gender_diffs_pre.append(np.abs(femaleness_scores[-1] - gender_forward))
            gender_index = attr_list_reverse.index('gender')
            gender_original_diffs.append(np.abs(femaleness_scores[gender_index+1] - femaleness_scores[gender_index]))
        if 'race' in attr_list:
            faces_race_diffs_pre.append(np.abs(race_predictions[-1] - race_forward))
            race_index = attr_list_reverse.index('race')
            race_original_diffs.append(np.abs(race_predictions[race_index+1] - race_predictions[race_index]))
        if 'smile' in attr_list:
            faces_smile_diffs_pre.append(np.abs(celeba_attr['Smiling'][-1] - smile_forward))
            smile_index = attr_list_reverse.index('smile')
            smile_original_diffs.append(np.abs(celeba_attr['Smiling'][smile_index+1] - celeba_attr['Smiling'][smile_index]))
        if 'bangs' in attr_list:
            faces_bangs_diffs_pre.append(np.abs(celeba_attr['Bangs'][-1] - bangs_forward))
            bangs_index = attr_list_reverse.index('bangs')
            bangs_original_diffs.append(np.abs(celeba_attr['Bangs'][bangs_index+1] - celeba_attr['Bangs'][bangs_index]))
        if 'beard' in attr_list:
            faces_beard_diffs_pre.append(np.abs(celeba_attr['No_Beard'][-1] - beard_forward))
            beard_index = attr_list_reverse.index('beard')
            beard_original_diffs.append(np.abs(celeba_attr['No_Beard'][beard_index+1] - celeba_attr['No_Beard'][beard_index]))
        if 'yaw' in attr_list:
            faces_yaw_diffs_pre.append(np.abs(yaw_predicted.cpu().numpy()[-1] - yaw_forward))
            yaw_index = attr_list_reverse.index('yaw')
            yaw_original_diffs.append(np.abs(yaw_predicted.cpu().numpy()[yaw_index+1] - yaw_predicted.cpu().numpy()[yaw_index]))
        if 'roll' in attr_list:
            faces_roll_diffs_pre.append(np.abs(roll_predicted.cpu().numpy()[-1] - roll_forward))
            roll_index = attr_list_reverse.index('roll')
            roll_original_diffs.append(np.abs(roll_predicted.cpu().numpy()[roll_index+1] - roll_predicted.cpu().numpy()[roll_index]))
        if 'pitch' in attr_list:
            faces_pitch_diffs_pre.append(np.abs(pitch_predicted.cpu().numpy()[-1] - pitch_forward))
            pitch_index = attr_list_reverse.index('pitch')
            pitch_original_diffs.append(np.abs(pitch_predicted.cpu().numpy()[pitch_index+1] - pitch_predicted.cpu().numpy()[pitch_index]))
        # Empty CUDA cache
        if use_cuda:
            torch.cuda.empty_cache()

        if args.verbose:
            update_stdout(1)
        # ============================================================================================================ #

        # Save all latent paths and shifts for the current latent code (sample) in a tensor of size:
        #   paths_latent_codes : torch.Size([num_gen_paths, 2 * args.shift_steps + 1, G.dim_z])
        # torch.save(torch.cat(paths_latent_codes), osp.join(latent_code_dir, 'paths_latent_codes.pt'))

    print()

    txt_out_dir = os.path.join(out_dir, "noncycle_commutative_score_attr_")
    txt_out_dir = txt_out_dir + 'age_' if 'age' in attr_list else txt_out_dir
    txt_out_dir = txt_out_dir + 'gender_' if 'gender' in attr_list else txt_out_dir
    txt_out_dir = txt_out_dir + 'race_' if 'race' in attr_list else txt_out_dir
    txt_out_dir = txt_out_dir + 'smile_' if 'smile' in attr_list else txt_out_dir
    txt_out_dir = txt_out_dir + 'bangs_' if 'bangs' in attr_list else txt_out_dir
    txt_out_dir = txt_out_dir + 'beard_' if 'beard' in attr_list else txt_out_dir
    txt_out_dir = txt_out_dir + 'yaw_' if 'yaw' in attr_list else txt_out_dir
    txt_out_dir = txt_out_dir + 'roll_' if 'roll' in attr_list else txt_out_dir
    txt_out_dir = txt_out_dir + 'pitch_' if 'pitch' in attr_list else txt_out_dir
    txt_out_dir = txt_out_dir + '.txt'
    with open(txt_out_dir, 'w') as f:
        if 'age' in attr_list:
            age_index = attr_list.index('age')
            print('age: {}'.format(dim[age_index]), file=f)
            print('age commutative score: {}, list is {}'.format(np.mean(faces_age_diffs_pre), faces_age_diffs_pre), file=f)
        if 'gender' in attr_list:
            gender_index = attr_list.index('gender')
            print('gender: {}'.format(dim[gender_index]), file=f)
            print('gender commutative score: {}, list is {}'.format(np.mean(faces_gender_diffs_pre), faces_gender_diffs_pre), file=f)
        if 'race' in attr_list:
            race_index = attr_list.index('race')
            print('race: {}'.format(dim[race_index]), file=f)
            print('race commutative score: {}, list is {}'.format(np.mean(faces_race_diffs_pre), faces_race_diffs_pre), file=f)
        if 'smile' in attr_list:
            smile_index = attr_list.index('smile')
            print('smile: {}'.format(dim[smile_index]), file=f)
            print('smile commutative score: {}, list is {}'.format(np.mean(faces_smile_diffs_pre), faces_smile_diffs_pre), file=f)
        if 'bangs' in attr_list:
            bangs_index = attr_list.index('bangs')
            print('bangs: {}'.format(dim[bangs_index]), file=f)
            print('bangs commutative score: {}, list is {}'.format(np.mean(faces_bangs_diffs_pre), faces_bangs_diffs_pre), file=f)
        if 'beard' in attr_list:
            beard_index = attr_list.index('beard')
            print('beard: {}'.format(dim[beard_index]), file=f)
            print('beard commutative score: {}, list is {}'.format(np.mean(faces_beard_diffs_pre), faces_beard_diffs_pre), file=f)
        if 'yaw' in attr_list:
            yaw_index = attr_list.index('yaw')
            print('yaw: {}'.format(dim[yaw_index]), file=f)
            print('yaw commutative score: {}, list is {}'.format(np.mean(faces_yaw_diffs_pre), faces_yaw_diffs_pre), file=f)
        if 'roll' in attr_list:
            roll_index = attr_list.index('roll')
            print('roll: {}'.format(dim[roll_index]), file=f)
            print('roll commutative score: {}, list is {}'.format(np.mean(faces_roll_diffs_pre), faces_roll_diffs_pre), file=f)
        if 'pitch' in attr_list:
            pitch_index = attr_list.index('pitch')
            print('pitch: {}'.format(dim[pitch_index]), file=f)
            print('pitch commutative score: {}, list is {}'.format(np.mean(faces_pitch_diffs_pre), faces_pitch_diffs_pre), file=f)

if __name__ == '__main__':
    main()
