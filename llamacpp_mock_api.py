from typing import Optional

import fire
from flask import Flask, jsonify, request
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
):
    # Create our Code Llama object.
    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
    )

    # With torchrun and distributed PyTorch, multiple copies of this code
    # can be run at once. We only want one of them (node 0) to have the Flask API
    # and we will use it to control the rest.
    if dist.get_rank() == 0:
        app = Flask(__name__)
        
        def run_chat_completion(instructions):
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

        @app.route("/v1/completions", methods=["POST"])
        def completions():
            content = request.json
            
            # Is used by Continue to generate a relevant title corresponding to the
            # model's response, however, the current prompt passed by Continue is not
            # good at obtaining a title from Code Llama's completion feature so we
            # use chat completion instead.
            messages = [
                {
                    "role": "user",
                    "content": content["prompt"]
                }
            ]
            
            # Perform Code Llama chat completion.
            response = run_chat_completion([messages])
            
            # Send back the response.
            return jsonify({"choices": [{"text": response}]})

        @app.route("/v1/chat/completions", methods=["POST"])
        def chat_completions():
            content = request.json
            messages = content["messages"]
            
            # Continue does not follow the user-assistant turn constraints Code Llama
            # needs. It has duplicate subsequent responses for a role. For example, a/u/u/a
            # will be sent by Continue when Code Llama only supports u/a/u/a so we squash
            # duplicate subsequent roles into a single message.
            if messages[0]["role"] == "assistant":
                messages[0]["role"] = "system"
            last_role = None
            remove_elements = []
            for i in range(len(messages)):
                if messages[i]["role"] == last_role:
                    messages[i-1]["content"] += "\n\n" + messages[i]["content"]
                    remove_elements.append(i)
                else:
                    last_role = messages[i]["role"]
            for element in remove_elements:
                messages.pop(element)

            # Perform Code Llama chat completion.
            response = run_chat_completion([messages])

            # Send JSON with Code Llama's response back to the VSCode Continue
            # extension. Note the extension expects six characters preappended to the
            # reponse JSON so we preappend the random string "onesix" to fulfill that requirement.
            return "onesix" + jsonify({"choices": [{"delta": {"role": "assistant", "content": response}}]}).get_data(as_text=True)

        # Run the Flask API server.
        app.run(port=8000)
    
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
