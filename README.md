## Setup
```bash
git clone https://github.com/Yingziiiiii-icra/ULR2SS.git 
cd ULR2SS
conda create -n ULR2SS python=3.8
conda activte ULR2SS
pip install -r requirements.txt
```

## Weight
For convenience, our pre-trained ULR2SS model can be downloaded directly here:
[ULR2SS_Weight](https://drive.google.com/file/d/1QhA2XHYmiajAhTJt9WqJocHGk6vEq3Tj/view)

## Demo Test
```bash
python inference.py \
  --input  /home/user/ULR2SS/images/rgb_demo1.png \      # path to image/folder
  --output /home/user/ULR2SS/images/output \  # path to save results
  --checkpoint /home/user/ULR2SS/joint_checkpoint_best.pth \ # ckpt path
```
