import os

import gradio as gr
import nltk
import openai

openai_engines = ["text-davinci-003", "code-davinci-002", "text-curie-001"]
prompt = "The following is a conversation with an AI assistant. The assistant is helpful, creative, clever, and very friendly.\n\nHuman: Hello, who are you?\nAI: I am an AI created by OpenAI. How can I help you today?"


def openai_completion(
    prompt,
    openai_token=None,
    engine="text-davinci-003",
    temperature=0.9,
    max_tokens=150,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0.6,
    stop=[" Human:", " AI:"],
):
    openai.api_key = openai_token
    response = openai.Completion.create(
        engine=engine,
        prompt=prompt,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
        frequency_penalty=frequency_penalty,
        presence_penalty=presence_penalty,
        stop=stop,
    )
    return response.choices[0].text


def chatgpt3(
    prompt,
    history,
    openai_token,
    engine,
    temperature,
    max_tokens,
    top_p,
    frequency_penalty,
    presence_penalty,
):
    history = history or []
    history_prompt = list(sum(history, ()))
    history_prompt.append(f"\nHuman: {prompt}")
    inp = " ".join(history_prompt)

    # keep the prompt length limited to ~2000 tokens
    inp = " ".join(inp.split()[-2000:])

    # remove duplicate sentences
    sentences = nltk.sent_tokenize(inp)
    sentence_dict = {}
    for i, s in enumerate(sentences):
        if s not in sentence_dict:
            sentence_dict[s] = i

    unique_sentences = [sentences[i] for i in sorted(sentence_dict.values())]
    inp = " ".join(unique_sentences)

    # create the output with openai
    out = openai_completion(
        inp,
        openai_token,
        engine,
        temperature,
        max_tokens,
        top_p,
        frequency_penalty,
        presence_penalty,
    )
    history.append((inp, out))
    return history, history, ""


with gr.Blocks(title="Chat with GPT-3") as block:
    gr.Markdown("## Chat with GPT-3")
    with gr.Row():
        with gr.Column():
            openai_token = gr.Textbox(label="OpenAI API Key", value=os.getenv("OPENAI_API_KEY"))
            engine = gr.Dropdown(
                label="GPT3 Engine",
                choices=openai_engines,
                value="text-davinci-003",
            )
            temperature = gr.Slider(label="Temperature", minimum=0, maximum=1, step=0.1, value=0.9)
            max_tokens = gr.Slider(label="Max Tokens", minimum=10, maximum=400, step=10, value=150)
            top_p = gr.Slider(label="Top P", minimum=0, maximum=1, step=0.1, value=1)
            frequency_penalty = gr.Slider(
                label="Frequency Penalty",
                minimum=0,
                maximum=1,
                step=0.1,
                value=0,
            )
            presence_penalty = gr.Slider(
                label="Presence Penalty",
                minimum=0,
                maximum=1,
                step=0.1,
                value=0.6,
            )

        with gr.Column():
            chatbot = gr.Chatbot()
            message = gr.Textbox(value=prompt, label="Type your question here:")
            state = gr.State()
            message.submit(
                fn=chatgpt3,
                inputs=[
                    message,
                    state,
                    openai_token,
                    engine,
                    temperature,
                    max_tokens,
                    top_p,
                    frequency_penalty,
                    presence_penalty,
                ],
                outputs=[chatbot, state, message],
            )
            submit = gr.Button("Send")
            submit.click(
                chatgpt3,
                inputs=[
                    message,
                    state,
                    openai_token,
                    engine,
                    temperature,
                    max_tokens,
                    top_p,
                    frequency_penalty,
                    presence_penalty,
                ],
                outputs=[chatbot, state, message],
            )

if __name__ == "__main__":
    block.launch(debug=True)
