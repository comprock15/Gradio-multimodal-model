# Gradio-multimodal-model

This repository demonstrates Gradio web-interface for multimodal model [SmolVLM2](https://huggingface.co/blog/smolvlm2).

Model use-cases:
- Image description
- Optical character recognition
- Video description

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
bash docker/start.sh
```

If you start you container for the first time, the webpage can be unavailable for some time as models are not downloaded yet. Wait for a bit and refresh the page.

## How to configure parameters

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