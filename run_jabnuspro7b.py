import os
import json
from pathlib import Path
from transformers import AutoModelForVision2Seq, AutoProcessor
import torch
from PIL import Image
from tqdm import tqdm

# 加载 Janus-Pro-7B 模型和处理器
model = AutoModelForVision2Seq.from_pretrained(
    "janus-project/Janus-Pro-7B",
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
processor = AutoProcessor.from_pretrained("janus-project/Janus-Pro-7B")

# 图像目录
image_dir = Path("./dataset/image")
assert image_dir.exists(), f"Directory not found: {image_dir}"

# 输出路径
output_file = Path("run_janus_results.json")

# 读取 prompt
with open("prompt.txt", "r", encoding="utf-8") as f:
    prompt_text = f.read()

# 批处理图片
results = {}
image_files = list(image_dir.glob("*.[jp][pn]g"))  # 支持 jpg 和 png

for image_path in tqdm(image_files, desc="Processing Images"):
    try:
        image = Image.open(image_path).convert("RGB")

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

        try:
            result_json = json.loads(output_text)
        except json.JSONDecodeError:
            result_json = {"error": "Failed to parse output as JSON", "raw": output_text}

        results[image_path.name] = result_json
        print(f"✅ Processed: {image_path.name}")
    except Exception as e:
        results[image_path.name] = {"error": str(e)}
        print(f"❌ Failed: {image_path.name} | {e}")

# 保存结果
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print(f"\n🎉 All results saved to: {output_file.resolve()}")
