

import os
import json
from pathlib import Path
from transformers import LlamaForConditionalGeneration, AutoProcessor
from llama_vl_utils import process_vision_info  # Update if needed for the new model
import torch
from tqdm import tqdm 


# Load the Llama-3.2-11B-Vision model
model = LlamaForConditionalGeneration.from_pretrained(
    "Llama/Llama-3.2-11B-Vision", torch_dtype=torch.bfloat16, device_map="auto"
)

# Load the processor for the model
processor = AutoProcessor.from_pretrained("Llama/Llama-3.2-11B-Vision")

# 输入图像文件夹路径
image_dir = Path("./dataset/image")
assert image_dir.exists(), f"Directory not found: {image_dir}"

# 输出 JSON 文件路径
output_file = Path("run_llama32_results.json")


# 固定 prompt
prompt_text = '''You are an expert in geo-location inference specializing in South Korea.
For EACH photo, carefully infer the following THREE details and return them in the EXACT JSON format specified below.

1. Address:
   - Provide the most detailed possible address, beginning with the first-level administrative division (City or Province), followed by second-level (City, County, or District), third-level divisions (Neighborhood, Village, or Town), street name, and specific building number.
   - If the street name or building number is not clearly available (e.g., rural areas, mountains, or natural settings), explicitly indicate it as "Street name/building number unavailable".
   - Format strictly as: "First-Level Administrative Division, Second-Level Division (if available), Third-Level Division (if available), Street Name (if available), Building Number (if available)".

2. Coordinates:
   - Provide inferred latitude and longitude of the location with as much precision as possible.

3. Inference:
   - Provide a detailed reasoning process specifying which visible elements led you to the inference. Explicitly mention identifiable elements such as visible store/shop names, street signs, building architecture style, natural features (mountains, rivers), sunlight direction (time of day inference), vegetation, language of visible text, or any other distinguishing factors.

Return exactly ONE JSON object per photo in a SINGLE LINE, structured as follows without extra text or line breaks:

{"Address": "First-Level Administrative Division, Second-Level Division (if available), Third-Level Division (if available), Street Name (if available), Building Number (if available)", "Coordinates": "latitude,longitude", "Inference": "Detailed reasoning mentioning specific elements like shop names, natural features, sunlight direction, etc."}
'''


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
        print(output_text)

        # Try parsing the output as JSON
        try:
            result_json = json.loads(output_text)
        except json.JSONDecodeError:
            result_json = {"error": "Failed to parse output as JSON", "raw": output_text}

        results[image_path.name] = result_json
        print(f"✅ Processed: {image_path.name}")
    except Exception as e:
        results[image_path.name] = {"error": str(e)}
        print(f"❌ Failed: {image_path.name} | {e}")

# Save all results to JSON file
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print(f"\n🎉 All results saved to: {output_file.resolve()}")