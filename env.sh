# LIB
rsync -av --progress /gemini/data-1/lm-evaluation-harness/ ./lm-evaluation-harness/
rsync -av --progress /gemini/data-1/LLaMA-Factory/ ./LLaMA-Factory/
pip install -e /gemini/code/lm-evaluation-harness
pip install -e /gemini/code/LLaMA-Factory[torch,metrics]

# REF
git clone https://hub.gitmirror.com/https://github.com/luchaoshi45/llm.git
cd llm