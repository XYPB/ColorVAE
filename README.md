# Color VAE

### VAE package

Please install [survae](https://github.com/didriknielsen/survae_flows) first
```bash
pip install git+https://github.com/didriknielsen/survae_flows.git
```

### Downlaod Dataset

```bash
kaggle datasets download -d akash2sharma/tiny-imagenet
```

Please delete the duplicated folder in the zip file before training 

### Train

```bash
python main.py --using_vae --img_size 64 --dataset tinyImgNetZip --vis_mode tensorboard
```

