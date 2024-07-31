# Code Llama for VSCode

An API which mocks [Llama.cpp](https://github.com/ggerganov/llama.cpp) to enable support for Code Llama with the
[Continue Visual Studio Code extension](https://continue.dev/).

As of the time of writing and to my knowledge, this is the only way to use Code Llama with VSCode locally without having
to sign up or get an API key for a service. The only exception to this is Continue with [Ollama](https://ollama.ai/), but
Ollama doesn't support Windows or Linux. On the other hand, Code Llama for VSCode is completely cross-platform and will
run wherever Meta's own [codellama](https://github.com/facebookresearch/codellama) code will run.

Now let's get started!

### Setup

Prerequisites:
- [Download and run one of the Code Llama Instruct models](https://github.com/facebookresearch/codellama)
- [Install the Continue VSCode extension](https://marketplace.visualstudio.com/items?itemName=Continue.continue)

After you are able to use both independently, we will glue them together with Code Llama for VSCode.

Steps:
1. Move `llamacpp_mock_api.py` to your [`codellama`](https://github.com/facebookresearch/codellama) folder and install Flask to your environment with `pip install flask`.
2. Run `llamacpp_mock_api.py` with your [Code Llama Instruct torchrun command](https://github.com/facebookresearch/codellama#fine-tuned-instruction-models). For example:
```
torchrun --nproc_per_node 1 llamacpp_mock_api.py \
    --ckpt_dir CodeLlama-7b-Instruct/ \
    --tokenizer_path CodeLlama-7b-Instruct/tokenizer.model \
    --max_seq_len 512 --max_batch_size 4
```
3. Click the settings button at the bottom right of Continue's UI in VSCode and make changes to `config.json` so it looks like [this](https://docs.continue.dev/reference/Model%20Providers/llamacpp)[<sup>\[archive\]</sup>](http://web.archive.org/web/20240531162330/https://docs.continue.dev/reference/Model%20Providers/llamacpp). Replace `MODEL_NAME` with `codellama-7b`.

Restart VSCode or reload the Continue extension and you should now be able to use Code Llama for VSCode!
