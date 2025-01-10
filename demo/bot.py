from transformers import AutoModelForCausalLM, AutoTokenizer
from lm_eval.models.huggingface import HFLM
import torch

class QABot:
    def __init__(self, model_path, device="cuda"):
        """
        初始化问答机器人
        :param model_path: 模型路径
        :param device: 运行设备（如 "cuda" 或 "cpu"）
        """
        self.device = device
        self.model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", trust_remote_code=True).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.lm = HFLM(pretrained=self.model, tokenizer=self.tokenizer, batch_size=1, device=self.device)
        self.history = []  # 记录对话历史

    def ask(self, question, max_length=100):
        """
        向机器人提问
        :param question: 用户输入的问题
        :param max_length: 生成回答的最大长度
        :return: 模型的回答
        """
        inputs = self.tokenizer(question, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model.generate(
                inputs.input_ids,
                max_length=max_length,
                pad_token_id=self.tokenizer.eos_token_id
            )
        answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return answer
    
    def ask_history(self, question, max_length=100):
        # 将历史记录和当前问题拼接
        context = "\n".join(self.history + [f"用户: {question}"])
        inputs = self.tokenizer(context, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model.generate(
                inputs.input_ids,
                max_length=max_length,
                pad_token_id=self.tokenizer.eos_token_id
            )
        answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        self.history.append(f"用户: {question}")  # 更新历史记录
        self.history.append(f"机器人: {answer}")
        return answer

    def evaluate(self, task_name):
        """
        评估模型在指定任务上的表现
        :param task_name: 任务名称（如 "arc_challenge"）
        :return: 评估结果
        """
        from lm_eval import simple_evaluate
        results = simple_evaluate(model=self.lm, tasks=[task_name])
        return results