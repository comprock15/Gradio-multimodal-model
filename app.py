from transformers import pipeline
import gradio as gr
import torch
import threading
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

device_lock = threading.Lock() # to make operations including device safe

pipe = pipeline(
    "image-text-to-text",
    model="HuggingFaceTB/SmolVLM2-256M-Video-Instruct",
    dtype=torch.float16,
    device='cpu'
)

system_prompt = dict() # stores last media sent so that user doesn't have to load it again

# supported file formats
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
        #pass
    logger.info('Device lock released')

    if media:
        update_last_media(media)

    logger.info(f'Generated text: {result[0]['generated_text']}')
    return result[0]['generated_text'][-1]['content']
    # return 'ok'

def device_change(value):
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

    return gr.Radio(label='Device', choices=['cpu', 'cuda'], value=value, interactive=True)

demo = gr.Blocks(theme=gr.themes.Ocean())

with demo:
    gr.Header("SmolVLM2")

    device_radio = gr.Radio(label='Device', choices=['cpu', 'cuda'], value='cpu', interactive=True)
    device_radio.input(fn=device_change, inputs=device_radio, outputs=device_radio)

    gr.ChatInterface(
        fn=answer,
        type="messages",
        title="SmolVLM2 Bot",
        textbox=gr.MultimodalTextbox(
            file_types=image_type + video_type,
            file_count='multiple',
        ),
        multimodal=True,
        examples = [
            {'text': 'Describe the image', 'files': ['https://www.hshv.org/wp-content/uploads/2020/09/GettyImages-1152049636.jpg']},
            {'text': 'Describe the video', 'files': ['https://media.roboflow.com/supervision/video-examples/croissant-1280x720.mp4']},
        ]
    )

demo.launch(
    share=True,
    debug=True,
    max_file_size=15 * gr.FileSize.MB
)
