

from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch

# default: Load the model on the available device(s)
model = Qwen2VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2-VL-7B-Instruct", torch_dtype=torch.bfloat16, device_map="auto"
)

# We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.
# model = Qwen2VLForConditionalGeneration.from_pretrained(
#     "Qwen/Qwen2-VL-7B-Instruct",
#     torch_dtype=torch.bfloat16,
#     attn_implementation="flash_attention_2",
#     device_map="auto",
# )

# default processer
processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")

# The default range for the number of visual tokens per image in the model is 4-16384. You can set min_pixels and max_pixels according to your needs, such as a token count range of 256-1280, to balance speed and memory usage.
# min_pixels = 256*28*28
# max_pixels = 1280*28*28
# processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct", min_pixels=min_pixels, max_pixels=max_pixels)

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": "file://./dataset/image/C0_COM_001.jpg",
            },
            {"type": "text", "text": '''You are an expert in geo-location inference. 
                For EACH photo, infer the following THREE pieces of information based on the location being within South Korea, and return them in a specific format:

                1. Address        : [the inferred address of the image location, starting with the first-level administrative division in South Korea (e.g., City or Province), followed by second-level and third-level administrative divisions (if applicable), then the street name and building number]
                - If any administrative division (second-level or third-level) is not available, leave it blank and format the address as "First-Level Administrative Division, Street Name, Building Number".
                - Try to infer as much detail as possible, including the second and third-level administrative divisions.
                2. Coordinates    : [the inferred latitude and longitude of the location]
                3. Inference      : [the reasoning process that led to this address and coordinates]

                Return a SINGLE LINE JSON with EXACTLY these keys and no extra text. Format it as:
                {"Address": "First-Level Administrative Division, Second-Level Administrative Division (if applicable), Third-Level Administrative Division (if applicable), Street Name, Building Number", "Coordinates": "latitude,longitude", "Inference": "reasoning_process_here"}'''
             },
        ],
    }
]

# Preparation for inference
text = processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
image_inputs, video_inputs = process_vision_info(messages)
inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt",
)
inputs = inputs.to("cuda")

# Inference: Generation of the output
generated_ids = model.generate(**inputs, max_new_tokens=128)
generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)
print(output_text)
