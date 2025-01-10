from transformers import AutoModelForCausalLM, AutoTokenizer
from lm_eval.models.huggingface import HFLM
from lm_eval import simple_evaluate
from lm_eval import tasks
import torch
import os, json 

# 强制指定设备为 CPU
device = torch.device("cpu")

# 加载模型和分词器
model = AutoModelForCausalLM.from_pretrained(
    _LLM_, 
    trust_remote_code=True
).to(device).eval()
tokenizer = AutoTokenizer.from_pretrained(_LLM_, trust_remote_code=True)

# 初始化 HFLM，并设置 device="cpu"
lm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=1, device="cpu")

# 运行评估任务
results = simple_evaluate(model=lm, tasks=["arc_challenge", "hellaswag", "piqa"])

# 保存结果到 JSON 文件
filtered_results = {key: value for key, value in results.items() if key == "results"}
os.makedirs("workdir", exist_ok=True)
with open("workdir/results.json", "w") as json_file:
    json.dump(filtered_results, json_file, indent=4)