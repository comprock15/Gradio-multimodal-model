# Gradio-multimodal-model

## Get started

1. Clone this repository
```
git clone https://github.com/comprock15/Gradio-multimodal-model.git

```
2. Go to the folder where you cloned this repository
```
cd Gradio-multimodal-model
```

3. Build docker image:
```
bash docker/build.sh
```

4. Start docker image:
```
bash docker.start.sh
```

## Configure parameters

You can configure some parameters in `docker/.env`:
```
# Container configuration
IMAGE_NAME=smolvlm2-app-image
CONTAINER_NAME=smolvlm2-app
MODEL_VOLUME=smolvlm2-models
CACHE_VOLUME=smolvlm2-cache
```

And some runtime parameters in `docker/start.sh`:
```
# Runtime configuration
DEVICE=${1:-cuda}     # cuda/cpu
MODEL_SIZE=${2:-256M} # 256M/500M
PORT=${3:-7865}       # any free port
```