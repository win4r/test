import torch
from spirit_gpu import start, Env
from transformers import MllamaForConditionalGeneration, AutoProcessor
from PIL import Image
import requests
from typing import Dict, Any
import io
import base64

model_path = "/workspace/llama3.2"
model = MllamaForConditionalGeneration.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
processor = AutoProcessor.from_pretrained(model_path)


def to_base64(image: Image.Image):
    im = image.convert("RGB")
    with io.BytesIO() as output:
        im.save(output, format="PNG")
        contents = output.getvalue()
        return base64.b64encode(contents).decode("utf-8")


def get_image_from_url_or_base64(image_data: str) -> Image.Image:
    if image_data.startswith('http://') or image_data.startswith('https://'):
        # It's a URL, download the image
        response = requests.get(image_data, stream=True)
        response.raise_for_status()
        return Image.open(io.BytesIO(response.content))
    else:
        # Assume it's a base64 string
        try:
            image_bytes = base64.b64decode(image_data)
            return Image.open(io.BytesIO(image_bytes))
        except:
            raise ValueError("Invalid image data. Must be a valid URL or base64 encoded image.")


def handler(request: Dict[str, Any], env: Env):
    print(f"Received request: {request}")
    input_data = request.get("input", {})

    # Get image data and text from input
    image_data = input_data.get("image")
    text = input_data.get("text")

    if not image_data or not text:
        return {"error": "Both 'image' and 'text' must be provided in the input."}

    # Get image from URL or base64
    try:
        image = get_image_from_url_or_base64(image_data)
    except Exception as e:
        return {"error": f"Failed to load image: {str(e)}"}

    messages = [
        {"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": text}
        ]}
    ]

    input_text = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(
        image,
        input_text,
        add_special_tokens=False,
        return_tensors="pt"
    ).to(model.device)

    output = model.generate(**inputs, max_new_tokens=30)
    result = processor.decode(output[0])

    # Convert image to base64 for response
    base64_image = to_base64(image)

    response = {
        "result": result,
        "image": base64_image
    }

    return response


start({"handler": handler})
