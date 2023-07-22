from potassium import Potassium, Request, Response
import torch
from transformers import pipeline
from transformers import AutoTokenizer, pipeline
from auto_gptq import AutoGPTQForCausalLM

app = Potassium("my_app")
model_name_or_path = "TheBloke/Llama-2-7b-Chat-GPTQ"
model_basename = "gptq_model-4bit-128g"
use_triton = False

# @app.init runs at startup, and loads models into the app's context
@app.init
def init():
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
    model = AutoGPTQForCausalLM.from_quantized(
        model_name_or_path,
        model_basename=model_basename,
        use_safetensors=True,
        trust_remote_code=True,
        device="cuda:0",
        use_triton=use_triton,
        quantize_config=None
    )
    context = {
        "model": model,
        "tokenizer": tokenizer
    }
    return context

# @app.handler runs for every call
@app.handler("/")
def handler(context: dict, request: Request) -> Response:
    prompt = request.json.get("prompt")
    tokenizer = context.get("tokenizer")
    model = context.get("model")

    prompt_template=f'''[INST] <<SYS>>
    You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature. If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.
    <</SYS>>
    {prompt}[/INST]'''

    print("-Pipeline-")
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=512,
        temperature=0.7,
        top_p=0.95,
        repetition_penalty=1.15
    )
    output = pipe(prompt_template)[0]['generated_text']

    return Response(
        json = {"outputs": output}, 
        status=200
    )

if __name__ == "__main__":
    app.serve()