# Make Container
## Download the container
- ref site
  - [enroot github](https://github.com/NVIDIA/enroot)
  - [NGC(Nvidia GPU Cloud) PyTorch container](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch/tags)

1. See what you want to download the version of enroot container.    
   example) Tgas V24.06-py3     
2. you install `enroot`
3. download specific version container.    
   `enroot import  --output nvcr.io+nvidia+pytorch+24.06-py3.sqsh  docker://nvcr.io/nvidia/pytorch:24.06-py3`

## Make `deepspeed.sqsh` container
1. enter container and install `deepspeed`
- pakages
  - `deepspeed`
  - `transformer`
