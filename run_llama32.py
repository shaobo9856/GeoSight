import os
import json
from pathlib import Path
from transformers import MllamaForConditionalGeneration, AutoProcessor
import torch
from tqdm import tqdm 
from PIL import Image

def try_fix_json(text):
    # 粗略修复不完整的 JSON 输出（主要用于缺少末尾括号）
    stack = []
    for c in text:
        if c == "{":
            stack.append("{")
        elif c == "}":
            if stack and stack[-1] == "{":
                stack.pop()
            else:
                return text  # 括号错误，放弃修复
    text += "}" * len(stack)  # 补全缺失的右括号
    return text

# Load the Llama-3.2-11B-Vision model
model = MllamaForConditionalGeneration.from_pretrained(
    "meta-llama/Llama-3.2-11B-Vision-Instruct", torch_dtype=torch.bfloat16, device_map="auto"
)

# Load the processor for the model
processor = AutoProcessor.from_pretrained("meta-llama/Llama-3.2-11B-Vision-Instruct")

# 输入图像文件夹路径
image_dir = Path("./dataset/image")
assert image_dir.exists(), f"Directory not found: {image_dir}"

# 输出 JSON 文件路径
output_file = Path("llama32_results.json")

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
image_files = list(image_dir.glob("*.[jp][pn]g"))  # 支持 jpg 和 png

# 允许跳过前 N 张（如果中断过）
START_INDEX = 0  # 可改成你上次中断的 index，比如 150

for idx, image_path in enumerate(tqdm(image_files, desc="Processing Images")):
    if idx < START_INDEX or image_path.name in results:
        continue
        
    try:
        image_inputs = Image.open(image_path)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt_text},
                ],
            }
        ]
        prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(
            text=[prompt],
            images=image_inputs,
            add_special_tokens=False,
            return_tensors="pt",
        ).to("cuda")

        generated_ids = model.generate(**inputs, max_new_tokens=384)
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
            fixed = try_fix_json(output_text)
            try:
                result_json = json.loads(fixed)
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

# Save all results to JSON file
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print(f"\n🎉 All results saved to: {output_file.resolve()}")

