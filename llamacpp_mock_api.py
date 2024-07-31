from typing import Optional

import fire
from flask import Flask, jsonify, request, Response
import torch.distributed as dist

from llama import Llama

def main(
    ckpt_dir: str,
    tokenizer_path: str,
    temperature: float = 0.2,
    top_p: float = 0.95,
    max_seq_len: int = 512,
    max_batch_size: int = 8,
    max_gen_len: Optional[int] = None,
    port: int = 8080,
):
    print("Loading Code Llama...", end="", flush=True)

    # Create our Code Llama object.
    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
    )
    
    print("Done!", flush=True)
    print()

    # With torchrun and distributed PyTorch, multiple copies of this code
    # can be run at once. We only want one of them (node 0) to have the Flask API
    # and we will use it to control the rest.
    if dist.get_rank() == 0:
        app = Flask(__name__)
        
        def prompt_to_instructions(prompt):
            # Remove unnecessary tokens and spacing from Continue's prompt format.
            prompt = prompt.replace("</s>\n<s>", "")
            prompt = prompt.replace("[INST] ", "[INST]")
            prompt = prompt.replace(" [/INST]", "[/INST]")

            # Consume Continue's prompt string and transform it into a list of
            # message dicts which contain role information.
            messages = []
            prompt_start = 0
            while True:
                user_message_start = prompt.find("[INST]", prompt_start) + 6
                user_message_end = prompt.find("[/INST]", prompt_start)
                assistant_message_end = prompt.find("[INST]", user_message_end)
                
                messages += [{"role": "user", "content": prompt[user_message_start:user_message_end]}]
                
                if assistant_message_end != -1:
                    messages += [{"role": "assistant", "content": prompt[user_message_end + 7:assistant_message_end]}]
                else:
                    break

                prompt_start = assistant_message_end
            
            # Send back the message instructions.
            return [messages]

        def run_chat_completion(prompt):
            # Transform the prompt format Continue uses into a list of
            # message dicts Code Llama supports.
            instructions = prompt_to_instructions(prompt)
            
            # Broadcast what should be processed to other nodes (acting as a C&C node).
            dist.broadcast_object_list([instructions, max_gen_len, temperature, top_p])

            # Start Code Llama inferencing.
            results = generator.chat_completion(
                instructions,
                max_gen_len=max_gen_len,
                temperature=temperature,
                top_p=top_p,
            )
            
            # Send the response back.
            return results[0]["generation"]["content"].strip()

        @app.route("/completion", methods=["POST"])
        def completion():
            content = request.json
            
            print("Incoming request: " + str(content))
            
            # Perform Code Llama chat completion.
            response = run_chat_completion(content["prompt"])
            response = jsonify({"content": response}).get_data(as_text=True)
            
            print("Outgoing response: " + str(response))
            
            # Llama.cpp's HTTP server uses Server-Sent Events to stream results to the client
            # so we reimplement it here, for a single event sent to Continue which contains
            # the entire Code Llama response.
            def generate():
                yield "data: " + response + "\n"
                yield "data: [DONE]\n"
            
            # Send back the response.
            return Response(generate())

        # Run the Flask API server on the Llama.cpp port.
        app.run(port=port)
    
    # Nodes which are not node 0 wait for tasks.
    else:
        while True:
            config = [None] * 4
            try:
                dist.broadcast_object_list(config)
                generator.chat_completion(
                    config[0], max_gen_len=config[1], temperature=config[2], top_p=config[3]
                )
            except:
                pass

if __name__ == "__main__":
    fire.Fire(main)
