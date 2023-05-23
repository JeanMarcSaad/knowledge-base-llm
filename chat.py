#!/usr/bin/env python3
from dotenv import load_dotenv
from langchain.chains import LLMChain
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.llms import LlamaCpp, GPT4All
from langchain.prompts import PromptTemplate
import os

load_dotenv()

model_type = os.environ.get('MODEL_TYPE')
model_path = os.environ.get('MODEL_PATH')
model_n_ctx = os.environ.get('MODEL_N_CTX')

from constants import CHROMA_SETTINGS

def main():
    # callbacks = []
    callbacks = [StreamingStdOutCallbackHandler()]
    # prompt_template = "What is a good name for a company that makes {product}?"
    prompt_template = """
You are an AI assistant that specializes in helping users. Given a user prompt, return an output that is adhering to the user's instructions.

User: {query}
Assistant: 
"""
    match model_type:
        case "LlamaCpp":
            llm = LlamaCpp(model_path=model_path, n_ctx=model_n_ctx, callbacks=callbacks, verbose=False,)
        case "GPT4All":
            llm = GPT4All(model=model_path, n_ctx=model_n_ctx, backend='gptj', callbacks=callbacks, verbose=False)
        case _default:
            print(f"Model {model_type} not supported!")
            exit

    while True:
        query = input("\nEnter a query: ")
        if query == "exit":
            break

        # Get the answer from the chain
        llm_chain = LLMChain(
            llm=llm,
            prompt=PromptTemplate.from_template(prompt_template)
        )
        res = llm_chain(query)
        answer = res["text"]

        # Print the result
        print("\n\n> Question:")
        print(query)
        print("\n> Answer:")
        print(answer)

if __name__ == "__main__":
    main()
