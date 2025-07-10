import os
import argparse
import torch
from PIL import Image
from torchvision import transforms
import numpy as np
import config
from esrgan import Generator
from modeling.deeplab import DeepLab
from utils2.metrics import Evaluator

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def preprocess_image(image_path):
    img = Image.open(image_path).convert("RGB")

    # center‐crop if larger than 384×384
    if img.width > 384 or img.height > 384:
        left = (img.width - 384) // 2
        top  = (img.height - 384) // 2
        img = img.crop((left, top, left + 384, top + 384))

    # if exactly 384×384, down→up sample
    if img.size == (384, 384):
        img = img.resize((16, 16), Image.BICUBIC)
        img = img.resize((96, 96), Image.BICUBIC)

    return transforms.ToTensor()(img).unsqueeze(0)  # 1×3×H×W

def postprocess_image(tensor):
    tensor = tensor.squeeze(0).cpu().detach()
    img = (tensor * 0.5 + 0.5).clamp(0,1)
    return transforms.ToPILImage()(img)

def strip_module_state_dict(sd):
    from collections import OrderedDict
    new_sd = OrderedDict()
    for k, v in sd.items():
        new_sd[k.replace('module.', '')] = v
    return new_sd

def load_models(checkpoint_path):
    gen = Generator(config.IMG_CHANNELS).to(device)
    seg = DeepLab(num_classes=14, backbone='resnet', output_stride=16,
                  sync_bn=None, freeze_bn=False).to(device)

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    gen.load_state_dict(strip_module_state_dict(ckpt['gen_state_dict']))
    seg.load_state_dict(strip_module_state_dict(ckpt['seg_state_dict']))

    gen.eval()
    seg.eval()
    return gen, seg

def infer_and_save(image_path, gen_model, seg_model, output_folder):
    inp = preprocess_image(image_path).to(device)

    with torch.no_grad():
        sr = gen_model(inp)
    hr_img = postprocess_image(sr)
    base_name = os.path.basename(image_path)
    hr_img.save(os.path.join(output_folder, base_name))

    with torch.no_grad():
        seg_logits = seg_model(sr)
        seg_pred = torch.argmax(seg_logits, dim=1).squeeze().cpu().numpy().astype(np.uint8)
    seg_img = Image.fromarray(seg_pred)
    seg_img.save(os.path.join(output_folder, f"seg_{base_name}"))

def evaluate(test_folder, output_folder, checkpoint_path, gt_folder=None):
    gen_model, seg_model = load_models(checkpoint_path)
    evaluator = Evaluator(num_class=14) if gt_folder else None
    os.makedirs(output_folder, exist_ok=True)

    for fname in sorted(os.listdir(test_folder)):
        if not fname.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue
        img_path = os.path.join(test_folder, fname)
        infer_and_save(img_path, gen_model, seg_model, output_folder)

        # optional evaluation
        if evaluator:
            gt_path = os.path.join(gt_folder, fname)
            if os.path.exists(gt_path):
                gt = np.array(Image.open(gt_path).convert('L')).astype(np.int64)
                seg_pred = np.array(Image.open(os.path.join(output_folder, f"seg_{fname}")))
                evaluator.add_batch(gt, seg_pred)

    if evaluator:
        print(f"mIoU: {evaluator.Mean_Intersection_over_Union():.4f}, "
              f"PA: {evaluator.Pixel_Accuracy():.4f}, "
              f"PA_class: {evaluator.Pixel_Accuracy_Class():.4f}, "
              f"FWIoU: {evaluator.Frequency_Weighted_Intersection_over_Union():.4f}")

def main():
    parser = argparse.ArgumentParser(description="SR + Segmentation Inference")
    parser.add_argument('--input', required=True,
                        help="Path to image file or folder of images")
    parser.add_argument('--output', required=True,
                        help="Folder to save SR and segmentation outputs")
    parser.add_argument('--checkpoint', required=True,
                        help="Path to joint_checkpoint_best.pth")
    parser.add_argument('--gt_folder', default=None,
                        help="(Optional) Path to ground‐truth folder for evaluation")
    args = parser.parse_args()

    if os.path.isfile(args.input):
        os.makedirs(args.output, exist_ok=True)
        gen_model, seg_model = load_models(args.checkpoint)
        infer_and_save(args.input, gen_model, seg_model, args.output)
    elif os.path.isdir(args.input):
        evaluate(args.input, args.output, args.checkpoint, args.gt_folder)
    else:
        raise ValueError(f"Input path {args.input} is neither a file nor a directory.")

if __name__ == "__main__":
    main()





    # test_folder           = "/home/huang/ULR2SS/custom_dataset"
    # output_folder         = "/home/huang/ULR2SS/pred_seg"
    # gt_folder             = "/home/huang/ULR2SS/combined_384_copy"
    # joint_checkpoint_path = "/home/huang/ULR2SS/joint_checkpoint_best.pth"