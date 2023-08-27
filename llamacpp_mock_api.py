import json

import fire
from flask import Flask, Response, jsonify, request, stream_with_context

from model import CodeLlamaHF


def main(
    model_id: str,
    temperature: float = 0.2,
    top_p: float = 0.95,
    max_seq_len: int = 2048,
    load_in_4bit: bool = False,
    port: int = 8000,
):
    # Create our Code Llama object.
    code_llama = CodeLlamaHF(model_id=model_id, load_in_4bit=load_in_4bit)

    app = Flask(__name__)

    @app.route("/v1/completions", methods=["POST"])
    def completions():
        content = request.json

        # Is used by Continue to generate a relevant title corresponding to the
        # model's response, however, the current prompt passed by Continue is not
        # good at obtaining a title from Code Llama's completion feature so we
        # use chat completion instead.

        # Perform Code Llama chat completion.
        response = code_llama.completion(
            content["prompt"],
            max_seq_len=max_seq_len,
            temperature=temperature,
            top_p=top_p,
        )

        # Send back the response.
        return jsonify({"choices": [{"text": "".join(list(response))}]})

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
        system_prompt = None
        if messages[0]["role"] == "system":
            system_prompt = messages[0]["content"]
            messages.pop(0)
        last_role = None
        remove_elements = []
        for i in range(len(messages)):
            if messages[i]["role"] == last_role:
                messages[i - 1]["content"] += "\n\n" + messages[i]["content"]
                remove_elements.append(i)
            else:
                last_role = messages[i]["role"]
        for element in remove_elements:
            messages.pop(element)
        message = messages.pop()["content"]
        chat_history = [
            (messages[i]["content"], messages[i + 1]["content"])
            for i in range(0, len(messages), 2)
        ]

        # Perform Code Llama chat completion.
        response = code_llama.chat(
            message,
            chat_history,
            system_prompt,
            max_seq_len=max_seq_len,
            temperature=temperature,
            top_p=top_p,
        )

        def generate():
            # Send JSON with Code Llama's response back to the VSCode Continue
            # extension. Note the extension expects six characters preappended to the
            # reponse JSON so we preappend the random string "onesix" to fulfill that requirement.
            for chunk in response:
                yield "onesix" + json.dumps(
                    {"choices": [{"delta": {"role": "assistant", "content": chunk}}]}
                )

        return Response(stream_with_context(generate()))

        # Run the Flask API server.

    app.run(port=port)


if __name__ == "__main__":
    fire.Fire(main)
