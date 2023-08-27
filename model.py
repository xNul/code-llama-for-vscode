# adapted from https://huggingface.co/spaces/codellama/codellama-13b-chat

from threading import Thread
from typing import Iterator

import torch
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    TextIteratorStreamer,
)


class CodeLlamaHF:
    def __init__(
        self, model_id="codellama/CodeLlama-13b-Instruct-hf", load_in_4bit=True
    ):
        config = AutoConfig.from_pretrained(model_id)
        config.pretraining_tp = 1
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            config=config,
            torch_dtype=torch.float16,
            load_in_4bit=load_in_4bit,
            device_map="auto",
            use_safetensors=False,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)

    @staticmethod
    def get_prompt(
        message: str, chat_history: list[tuple[str, str]], system_prompt: str
    ) -> str:
        texts = [f"<s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n"]
        # The first user input is _not_ stripped
        do_strip = False
        for user_input, response in chat_history:
            user_input = user_input.strip() if do_strip else user_input
            do_strip = True
            texts.append(f"{user_input} [/INST] {response.strip()} </s><s>[INST] ")
        message = message.strip() if do_strip else message
        texts.append(f"{message} [/INST]")
        return "".join(texts)

    def run(
        self,
        prompt: str,
        max_seq_len: int = 2048,
        temperature: float = 0.1,
        top_p: float = 0.9,
    ) -> Iterator[str]:
        inputs = self.tokenizer(
            [prompt], return_tensors="pt", add_special_tokens=False
        ).to("cuda")

        streamer = TextIteratorStreamer(
            self.tokenizer, timeout=10.0, skip_prompt=True, skip_special_tokens=True
        )
        generate_kwargs = dict(
            inputs,
            streamer=streamer,
            max_length=max_seq_len,
            do_sample=True,
            top_p=top_p,
            temperature=temperature,
            num_beams=1,
        )
        t = Thread(target=self.model.generate, kwargs=generate_kwargs)
        t.start()

        for text in streamer:
            yield text

    def completion(
        self,
        prompt: str,
        max_seq_len: int = 2048,
        temperature: float = 0.1,
        top_p: float = 0.9,
    ) -> Iterator[str]:
        yield from self.run(
            prompt=f"<s>[INST] {prompt} [/INST]",
            max_seq_len=max_seq_len,
            temperature=temperature,
            top_p=top_p,
        )

    def chat(
        self,
        message: str,
        chat_history: list[tuple[str, str]],
        system_prompt: str,
        max_seq_len: int = 2048,
        temperature: float = 0.1,
        top_p: float = 0.9,
    ) -> Iterator[str]:
        prompt = self.get_prompt(message, chat_history, system_prompt)
        yield from self.run(
            prompt=prompt,
            max_seq_len=max_seq_len,
            temperature=temperature,
            top_p=top_p,
        )
