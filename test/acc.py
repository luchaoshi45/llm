from transformers import AutoModelForCausalLM, AutoTokenizer 
from lm_eval.models.huggingface import HFLM
from lm_eval import simple_evaluate
from lm_eval import tasks
import os, json

# 加载模型和分词器
model = AutoModelForCausalLM.from_pretrained(_LLM_, device_map="auto", trust_remote_code=True).eval()
tokenizer = AutoTokenizer.from_pretrained(_LLM_, trust_remote_code=True)

# 确保 HFLM 的设备与模型一致（使用 GPU）
lm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=1, device="cuda")

# 执行评估任务
# results = simple_evaluate(model=lm, tasks=["arc_challenge", "hellaswag", "piqa"])
results = simple_evaluate(model=lm, tasks=["arc_challenge"])

# 将 results 中的数据导出到 JSON 文件中
filtered_results = {key: value for key, value in results.items() if key == "results"}
json_filtered_results = json.dumps(filtered_results, indent=4)

os.makedirs("workdir", exist_ok=True)
with open("workdir/results.json", "w") as json_file:
    json_file.write(json_filtered_results)
