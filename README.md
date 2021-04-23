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
python main.py --using_vae --img_size 64 --dataset tinyImgNetZip --vis_mode wandb
```

### PSNR

|ColorVAE|Non-VAE|ECCV16|SIGGRAPH17|w/o semantic pre-train|w/ VAE hint|
|:-:|:-:|:-:|:-:|:-:|:-:|
|26.0008|24.6531|21.9863|25.6168|21.5979|27.0091|

### Rating

|Non-VAE|ECCV16|SIGGRAPH17|w/ VAE hint|
|:-:|:-:|:-:|:-:|
|3.555|3.188|3.546|3.691|
