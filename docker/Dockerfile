FROM huggingface/transformers-pytorch-gpu:latest

SHELL [ "/bin/bash", "-c" ]
ENV DEBIAN_FRONTEND=noninteractive

# Only user args for build
ARG UID=1000
ARG GID=1000
ARG USER=docker_user

RUN useradd -m ${USER} --uid=${UID} \
    && usermod -s /bin/bash ${USER} \
    && usermod -a -G sudo ${USER}

# Установка Python 3.10
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    python3.10-venv \
    && rm -rf /var/lib/apt/lists/*

# Создаем симлинк python -> python3 (опционально)
RUN ln -s /usr/bin/python3 /usr/bin/python

# Проверяем установку
RUN python --version && pip --version

WORKDIR /home/${USER}/smolvml2
COPY ../requirements.txt /home/${USER}/smolvml2/

# Copy application code
COPY app /home/${USER}/smolvml2/app

# Copy media files to image
COPY media /home/${USER}/smolvml2/media

# Create directories for models and cache
RUN mkdir -p /home/${USER}/smolvlm2/models /home/${USER}/smolvlm2/cache

RUN pip install -r requirements.txt

# Set only cache environment variables
ENV USER=${USER}
ENV TRANSFORMERS_CACHE=/home/${USER}/smolvlm2/cache
ENV HUGGINGFACE_HUB_CACHE=/home/${USER}/smolvlm2/cache
ENV HF_HOME=/home/${USER}/smolvlm2/cache
ENV GRADIO_SERVER_NAME="0.0.0.0"
ENV MEDIA_DIR=/home/${USER}/smolvml2/media

# DEVICE, MODEL_SIZE, PORT will be set at runtime

EXPOSE 7860

RUN chown -R ${USER}:${USER} /home/${USER}
USER ${USER}

CMD ["python", "app/app.py"]