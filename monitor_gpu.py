import subprocess
import time
import re

output_file = 'gpu_usage.txt'
prev_gpu_memory_info = None

while True:
    # 运行nvidia-smi命令并获取输出
    result = subprocess.run(['nvidia-smi'], stdout=subprocess.PIPE)
    gpu_info = result.stdout.decode('utf-8')
    gpu_info = gpu_info[gpu_info.index('\n', gpu_info.index('\n') + 1):]

    # 使用正则表达式匹配显存信息
    memory_pattern = re.compile(r'(\d+)MiB / (\d+)MiB')
    memory_info = memory_pattern.findall(gpu_info)

    # 检查三个显卡的显存数据是否发生变化
    if memory_info and len(memory_info) >= 3:
        if memory_info != prev_gpu_memory_info:
            with open(output_file, 'a') as file:
                file.write(f"{memory_info[0][0]}MiB / {memory_info[0][1]}MiB, {memory_info[1][0]}MiB / {memory_info[1][1]}MiB, {memory_info[2][0]}MiB / {memory_info[2][1]}MiB\n")
            
            prev_gpu_memory_info = memory_info

    time.sleep(1) 