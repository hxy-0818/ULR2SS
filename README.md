## Setup
```bash
git clone https://github.com/Yingziiiiii-icra/ULR2SS.git 

cd ULR2SS

pip install -r requirements.txt

## Weight
For convenience, our pre-trained DiT models can be downloaded directly here:
[ULR2SS_Weight](https://drive.google.com/file/d/1QhA2XHYmiajAhTJt9WqJocHGk6vEq3Tj/view)

## Demo Test
python inference.py \
  --input  path/to/input.jpg       # path to image/folder
  --output path/to/output_folder   # path to save results
  --checkpoint joint_checkpoint_best.pth \ ckpt path
  --gt_folder path/to/gt_folder  # optional: if gt is provided
