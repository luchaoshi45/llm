# LLM

## ENV

### BASE
conda create --name llm --clone fremamba
python -c "import torch; print(torch.cuda.is_available())"
python -c "import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No GPU available')"
python -c "import torch; print(torch.version.cuda)"
python -c "import torch; print(torch.cuda.device_count())"
python -c "import torch; t = torch.tensor([1.0]); t = t.to('cuda') if torch.cuda.is_available() else t; print(t.device)"

### TOOL
git config --global user.name "MiraBit"
git config --global user.email "luchaoshi45@gmail.com"
apt update
apt install nload
apt-get update && apt-get install -y rsync
pip install transformer
pip install nvitop
pip install modelscope

### REF
git clone https://hub.gitmirror.com/https://github.com/luchaoshi45/llm.git
cd llm

git clone --depth 1 https://hub.gitmirror.com/https://github.com/EleutherAI/lm-evaluation-harness
cd lm-evaluation-harness
pip install -e .
cd ..

git clone --depth 1 https://hub.gitmirror.com/https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
pip install -e ".[torch,metrics]"
cd ..

git lfs install
git clone https://www.modelscope.cn/Qwen/Qwen2.5-0.5B-Instruct.git
git clone https://www.modelscope.cn/Qwen/Qwen2.5-7B-Instruct.git


## TEST
python test/acc.py

python test/monitor.py test/benchmark.py
python test/benchmark.py

cd LLaMA-Factory
CUDA_VISIBLE_DEVICES=0 GRADIO_SHARE=1 GRADIO_SERVER_PORT=7860 llamafactory-cli webui

python demo/main.py
python demo/ui.py