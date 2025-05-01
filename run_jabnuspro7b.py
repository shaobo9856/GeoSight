import os
import json
from pathlib import Path
from transformers import AutoModelForCausalLM
from janus.models import MultiModalityCausalLM, VLChatProcessor
from janus.utils.io import load_pil_images
import torch
from PIL import Image
from tqdm import tqdm

# 加载 Janus-Pro-7B 模型和处理器
model = AutoModelForCausalLM.from_pretrained(
    "janus-project/Janus-Pro-7B",
    trust_remote_code=True
)
model = model.to(torch.bfloat16).cuda().eval()

processor = VLChatProcessor.from_pretrained("janus-project/Janus-Pro-7B")
tokenizer = processor.tokenizer

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
        conversation = [
            {
                "role": "<|User|>",
                "content": f"<image_placeholder>\n{prompt_text}",
                "images": [f"file://{image_path.resolve()}"],
            }
        ]
        pil_images = load_pil_images(conversation)
        prepare_inputs = processor(conversations=conversation, images=pil_images, force_batchify=True
        ).to(model.device)

        inputs_embeds = model.prepare_inputs_embeds(**prepare_inputs)

        outputs = model.language_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=prepare_inputs.attention_mask,
            pad_token_id=tokenizer.eos_token_id,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            max_new_tokens=512,
            do_sample=False,
            use_cache=True,
        )

        output_text = tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)

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
