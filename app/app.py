import os
import sys
from transformers import pipeline
import gradio as gr
import torch
import threading
import logging

# Add the parent directory to Python path
#sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configuration from environment variables
DEVICE = os.getenv('DEVICE', 'cpu')
MODEL_SIZE = os.getenv('MODEL_SIZE', '256M')
PORT = int(os.getenv('PORT', '7860'))
MEDIA_DIR = os.getenv('MEDIA_DIR', '/home/docker_user/smolvlm2/media')

# Cache directories from environment
CACHE_DIR = os.getenv('TRANSFORMERS_CACHE', '/home/docker_user/smolvlm2/cache')
MODEL_DIR = os.getenv('MODEL_DIR', '/home/docker_user/smolvlm2/models')

# Ensure directories exist
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Model configuration
MODEL_CONFIGS = {
    '256M': "HuggingFaceTB/SmolVLM2-256M-Video-Instruct",
    '500M': "HuggingFaceTB/SmolVLM2-500M-Video-Instruct",
}

def get_model_name(size):
    return MODEL_CONFIGS.get(size, MODEL_CONFIGS['256M'])

device_lock = threading.Lock()

def initialize_pipeline(device=DEVICE, model_size=MODEL_SIZE):
    """Initialize the pipeline with specified device and model size"""
    model_name = get_model_name(model_size)

    # Auto-detect device if cuda requested but not available
    if device == 'cuda' and not torch.cuda.is_available():
        device = 'cpu'
        logger.warning("CUDA not available, falling back to CPU")

    logger.info(f"Loading model: {model_name}")
    logger.info(f"Device: {device}")
    logger.info(f"Model size: {model_size}")
    logger.info(f"Cache directory: {CACHE_DIR}")
    logger.info(f"Model directory: {MODEL_DIR}")

    torch_dtype = torch.float16 if device == 'cuda' else torch.float32

    try:
        pipe = pipeline(
            "image-text-to-text",
            model=model_name,
            torch_dtype=torch_dtype,
            device=device,
            cache_dir=CACHE_DIR,
            model_kwargs={'cache_dir': MODEL_DIR}
        )
        logger.info("Model loaded successfully")
        return pipe
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise

pipe = initialize_pipeline()

# Auto-detect device if cuda requested but not available
if DEVICE == 'cuda' and not torch.cuda.is_available():
    logger.warning("CUDA not available, falling back to CPU")
    DEVICE = 'cpu'

system_prompt = dict()  # stores last media sent so that user doesn't have to load it again

# Supported file formats
image_type = ['.png', '.jpg', '.jpeg']
video_type = ['.mp4']

def get_content_type(file):
    ext = '.' + file.split('.')[-1]
    if ext in image_type:
        return {'type': 'image', 'image': file}
    elif ext in video_type:
        return {'type': 'video', 'video': file}
    else:
        logger.error('Error while loading content')
        return {'type': 'text', 'text': 'Error while loading content'}

def format_message(message):
    role = 'user'
    content = [{'type': 'text', 'text': message['text']}]
    for file in message["files"]:
        content.append(get_content_type(file))
    formatted_message = {'role': role, 'content': content}
    logger.info(f'Formatted message: {formatted_message}')
    return formatted_message

def get_attached_media(message):
    return list(filter(lambda item: item['type'] != 'text', message['content']))

def update_last_media(media):
    global system_prompt
    system_prompt.clear()
    system_prompt = {'role': 'system', 'content': media}
    logger.info(f'Updated system prompt: {system_prompt}')

def answer(message, history):
    global pipe
    logger.info(f'Message: {message}')
    logger.info(f'History: {history}')

    formatted_message = format_message(message)
    media = get_attached_media(formatted_message)

    text = [formatted_message]
    if not media and system_prompt:
        text = [system_prompt] + text

    logger.info('Trying to acquire device lock...')
    with device_lock:
        result = pipe(text=text)
    logger.info('Device lock released')

    if media:
        update_last_media(media)

    logger.info(f'Generated text: {result[0]["generated_text"]}')
    return result[0]['generated_text'][-1]['content']

def device_change(value):
    global pipe
    global DEVICE

    can_change = True
    if value == 'cuda':
        if not torch.cuda.is_available():
            value = 'cpu'
            logger.warning(f"CUDA isn't available!")
            gr.Warning("CUDA isn't available!", duration=5)
            can_change = False

    if can_change:
        logger.info('Trying to acquire device lock...')
        with device_lock:
            pipe.model.to(value)
            pipe.device = torch.device(value)
        logger.info('Device lock released')

    DEVICE = value

    return gr.Radio(label='Device', choices=['cpu', 'cuda'], value=DEVICE, interactive=True)

def model_size_change(value):
    global pipe
    logger.info(f'Changing model size to: {value}')
    logger.info('Trying to acquire device lock...')
    with device_lock:
        # Reinitialize pipeline with new model size
        pipe = pipeline(
            "image-text-to-text",
            model=get_model_name(value),
            torch_dtype=torch.float16 if DEVICE == 'cuda' else torch.float32,
            device=DEVICE,
            cache_dir=CACHE_DIR,
            model_kwargs={'cache_dir': MODEL_DIR}
        )
    logger.info('Device lock released')

    return gr.Dropdown(label='Model Size', choices=list(MODEL_CONFIGS.keys()), value=value, interactive=True)

demo = gr.Blocks(theme=gr.themes.Ocean())

with demo:
    gr.Header("SmolVLM2")

    with gr.Row():
        device_radio = gr.Radio(label='Device', choices=['cpu', 'cuda'], value=DEVICE, interactive=True)
        model_size_dropdown = gr.Dropdown(
            label='Model Size',
            choices=list(MODEL_CONFIGS.keys()),
            value=MODEL_SIZE,
            interactive=True
        )

    device_radio.input(fn=device_change, inputs=device_radio, outputs=device_radio)
    model_size_dropdown.input(fn=model_size_change, inputs=model_size_dropdown, outputs=model_size_dropdown)

    # Примеры из локальных файлов в Docker образе
    examples_list = [
        {'text': 'Describe the image', 'files': [os.path.join(MEDIA_DIR, 'image.jpg')]},
        {'text': 'Extract text from the image', 'files': [os.path.join(MEDIA_DIR, 'image_with_text.png')]},
        {'text': 'Describe the video', 'files': [os.path.join(MEDIA_DIR, 'video.mp4')]},
    ]

    gr.ChatInterface(
        fn=answer,
        type="messages",
        title="SmolVLM2 Bot",
        textbox=gr.MultimodalTextbox(
            file_types=image_type + video_type,
            file_count='multiple',
        ),
        multimodal=True,
        examples=examples_list
    )

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=PORT,
        share=False,
        debug=True,
        max_file_size=15 * gr.FileSize.MB
    )