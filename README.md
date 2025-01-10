# INSTALKL
conda create --name llm --clone fremamba
python -c "import torch; print(torch.cuda.is_available())"
python -c "import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No GPU available')"
python -c "import torch; print(torch.version.cuda)"
python -c "import torch; print(torch.cuda.device_count())"
python -c "import torch; t = torch.tensor([1.0]); t = t.to('cuda') if torch.cuda.is_available() else t; print(t.device)"

# LIB
git clone https://github.com/luchaoshi45/llm.git
cd llm

git clone --depth 1 https://github.com/EleutherAI/lm-evaluation-harness
cd lm-evaluation-harness
pip install -e .

git lfs install
git clone https://www.modelscope.cn/Qwen/Qwen2.5-0.5B-Instruct.git

git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
pip install -e ".[torch,metrics]"



# TEST
python test/monitor.py test/benchmark.py
python test/benchmark.py

cd LLaMA-Factory
CUDA_VISIBLE_DEVICES=0 GRADIO_SHARE=1 GRADIO_SERVER_PORT=7860 llamafactory-cli webui