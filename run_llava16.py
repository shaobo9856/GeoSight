import os
import json
from pathlib import Path
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
import torch
from tqdm import tqdm
from PIL import Image
import re
import json

def extract_json_from_text(text):
    # 正则匹配首个大括号包围的 JSON 对象
    match = re.search(r'\{.*?\}', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return {"error": "Failed to parse JSON", "raw": match.group(0)}
    return {"error": "No JSON object found", "raw": text}

# 加载模型和 processor（确保你已下载/配置好权重）
model = LlavaNextForConditionalGeneration.from_pretrained("llava-hf/llava-v1.6-34b-hf", torch_dtype=torch.float16, low_cpu_mem_usage=True) 
model.to("cuda:0")
processor = LlavaNextProcessor.from_pretrained("llava-hf/llava-v1.6-34b-hf")

# 输入图像文件夹路径
image_dir = Path("./dataset/image")
assert image_dir.exists(), f"Directory not found: {image_dir}"

# 输出 JSON 文件路径
output_file = Path("run_llava_results.json")

# 读取已有结果（如果存在）
if output_file.exists():
    with open(output_file, "r", encoding="utf-8") as f:
        results = json.load(f)
else:
    results = {}

# 读取 prompt
with open("prompt.txt", "r", encoding="utf-8") as f:
    prompt_text = f.read()
    
# 遍历图像并推理
image_files = list(image_dir.glob("*.[jp][pn]g")) 

# 允许跳过前 N 张（如果中断过）
START_INDEX = 0  # 可改成你上次中断的 index，比如 150

for idx, image_path in enumerate(tqdm(image_files, desc="Processing Images")):
    if idx < START_INDEX or image_path.name in results:
        continue

    try:
        image = Image.open(image_path).convert("RGB")
        conversation = [
            {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt_text},
                {"type": "image"},
                ],
            },
        ]
        prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
        inputs = processor(images=image, text=prompt, return_tensors="pt").to("cuda:0")

        with torch.inference_mode():
            generated_ids = model.generate(**inputs, max_new_tokens=256)
        
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        print(output_text)

        # 解析为 JSON 格式（如果可能）
        try:
            result_json = extract_json_from_text(output_text)
            # result_json = json.loads(output_text)
        except json.JSONDecodeError:
            result_json = {"error": "Failed to parse output as JSON", "raw": output_text}

        results[image_path.name] = result_json
        print(f"✅ Processed: {image_path.name}")
    except Exception as e:
        results[image_path.name] = {"error": str(e)}
        print(f"❌ Failed: {image_path.name} | {e}")

    # 每 10 张保存一次
    if (idx + 1) % 10 == 0:
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"💾 Partial save at index {idx + 1}")

with open(output_file, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
print(f"\n🎉 All results saved to: {output_file.resolve()}")