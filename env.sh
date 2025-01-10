# TOOL
git config --global user.name "MiraBit"
git config --global user.email "luchaoshi45@gmail.com"
pip install nvitop
apt update
apt install nload

# REF
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