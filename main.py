

import os
import json
from pathlib import Path
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch

# default: Load the model on the available device(s)
model = Qwen2VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2-VL-7B-Instruct", torch_dtype=torch.bfloat16, device_map="auto"
)

# default processer
processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")

# 输入图像文件夹路径
image_dir = Path("./dataset/image")
assert image_dir.exists(), f"Directory not found: {image_dir}"

# 输出 JSON 文件路径
output_file = Path("geo_inference_results.json")


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

# 固定 prompt
prompt_text = '''You are an expert in geo-location inference. 
For EACH photo, infer the following THREE pieces of information based on the location being within South Korea, and return them in a specific format:

1. Address        : [the inferred address of the image location, starting with the first-level administrative division in South Korea (e.g., City or Province), followed by second-level and third-level administrative divisions (if applicable), then the street name and building number]
- If any administrative division (second-level or third-level) is not available, leave it blank and format the address as "First-Level Administrative Division, Street Name, Building Number".
- Try to infer as much detail as possible, including the second and third-level administrative divisions.
2. Coordinates    : [the inferred latitude and longitude of the location]
3. Inference      : [the reasoning process that led to this address and coordinates]

Return a SINGLE LINE JSON with EXACTLY these keys and no extra text. Format it as:
{"Address": "First-Level Administrative Division, Second-Level Administrative Division (if applicable), Third-Level Administrative Division (if applicable), Street Name, Building Number", "Coordinates": "latitude,longitude", "Inference": "reasoning_process_here"}'''


# 遍历图像并推理
results = {}
image_files = list(image_dir.glob("*.[jp][pn]g"))  # 支持 jpg 和 png
for image_path in tqdm(image_files, desc="Processing Images"):
    try:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": f"file://{image_path.resolve()}"},
                    {"type": "text", "text": prompt_text},
                ],
            }
        ]

        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda")

        generated_ids = model.generate(**inputs, max_new_tokens=256)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

        # 解析为 JSON 格式（如果可能）
        try:
            result_json = json.loads(output_text)
        except json.JSONDecodeError:
            result_json = {"error": "Failed to parse output as JSON", "raw": output_text}

        results[image_path.name] = result_json
        print(f"✅ Processed: {image_path.name}")
    except Exception as e:
        results[image_path.name] = {"error": str(e)}
        print(f"❌ Failed: {image_path.name} | {e}")

# 保存所有结果
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print(f"\n🎉 All results saved to: {output_file.resolve()}")