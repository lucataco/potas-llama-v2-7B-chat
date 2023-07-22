# This file runs during container build time to get model weights built into the container

model_name_or_path = "TheBloke/Llama-2-7b-Chat-GPTQ"
model_basename = "gptq_model-4bit-128g"
use_triton = False

from transformers import AutoTokenizer
from auto_gptq import AutoGPTQForCausalLM

def download_model():
    # do a dry run of loading the huggingface model, which will download weights
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
    model = AutoGPTQForCausalLM.from_quantized(
        model_name_or_path,
        model_basename=model_basename,
        use_safetensors=True,
        trust_remote_code=True,
        use_triton=use_triton,
        quantize_config=None
    )

if __name__ == "__main__":
    download_model()