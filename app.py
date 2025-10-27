from transformers import pipeline
import gradio as gr
import torch

pipe = pipeline("image-text-to-text", model="Qwen/Qwen3-VL-2B-Instruct")

def generate_text(image, prompt, device):
    messages = [
        {
            'role': 'user',
            'content': [
                {'type': 'image', 'image': image},
                {'type': 'text', 'text': prompt},
            ],
        }
    ]

    if device == 'cuda':
        if not torch.cuda.is_available():
            device = 'cpu'

    pipe.model.to(device)
    result = pipe(text=messages)

    return result[0]['generated_text'][1]['content'], f'Device: {device}'

demo = gr.Interface(
    fn=generate_text,
    inputs=[
        gr.Image(type="pil", label='Image'),
        gr.Textbox(label='Prompt', placeholder='Что изображено на картинке?'),
        gr.Radio(label='Device', choices=['cpu', 'cuda'], value='cpu')
        ],
    outputs=[gr.Textbox(label='Result', lines=5), gr.Textbox(label='Log', lines=1)],
    title="Image-Text to Text with Qwen3-VL-2B-Instruct",
    description="Upload an image and provide a prompt to see model's answer.",
    examples=[
         ["https://upload.wikimedia.org/wikipedia/commons/thumb/1/18/Dog_Breeds.jpg/330px-Dog_Breeds.jpg",
         'Какой породы собака на картинке?'],
         ["https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/hub/parrots.png",
         'Что изображено на картинке?'],
         ["https://image.civitai.com/xG1nkqKTMzGDvpLrqFT7WA/fb665d83-88e3-4192-baa1-91f60e586787/width=768,quality=90/00034-2227807581.jpeg",
         'Какое время суток на картинке?'],
         ["https://media.istockphoto.com/id/1442417585/ru/%D1%84%D0%BE%D1%82%D0%BE/%D1%87%D0%B5%D0%BB%D0%BE%D0%B2%D0%B5%D0%BA-%D0%BF%D0%BE%D0%BB%D1%83%D1%87%D0%B0%D0%B5%D1%82-%D0%BA%D1%83%D1%81%D0%BE%D1%87%D0%B5%D0%BA-%D1%81%D1%8B%D1%80%D0%BD%D0%BE%D0%B9-%D0%BF%D0%B8%D1%86%D1%86%D1%8B-%D0%BF%D0%B5%D0%BF%D0%BF%D0%B5%D1%80%D0%BE%D0%BD%D0%B8.jpg?s=612x612&w=0&k=20&c=iXyj27EYo3EKZuucF-Njy_Q04-fdDYnjv1FENubixHY=",
         'Как приготовить это блюдо?'],
    ]
)

demo.launch(share=True, debug=True)