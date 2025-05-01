import os
import json
from pathlib import Path
from transformers import LlavaForConditionalGeneration, AutoProcessor
import torch
from tqdm import tqdm

# 加载模型和 processor（确保你已下载/配置好权重）
model = LlavaForConditionalGeneration.from_pretrained(
    "llava-hf/llava-v1.6-yi-34b",
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
processor = AutoProcessor.from_pretrained("llava-hf/llava-v1.6-yi-34b")

# 输入图像文件夹路径
image_dir = Path("./dataset/image")
assert image_dir.exists(), f"Directory not found: {image_dir}"

# 输出 JSON 文件路径
output_file = Path("run_llava_results.json")

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
        # 读取图像并构造输入
        image = processor.image_processor.open(image_path)

        inputs = processor(
            text=prompt_text,
            images=image,
            return_tensors="pt"
        ).to(model.device)

        generated_ids = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=False
        )

        output_text = processor.batch_decode(
            generated_ids, skip_special_tokens=True
        )[0].strip()

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
