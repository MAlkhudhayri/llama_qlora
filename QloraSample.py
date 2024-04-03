from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer, LlamaTokenizer
import torch
import json
import os
tokenizer = LlamaTokenizer.from_pretrained('/hdd4/zoo/llama2/llama2-7b-hf')
batch_size = 4
input_ids = tokenizer.encode('os:', return_tensors='pt')
# tokenizer(
#     # batch_size*[''],
#     ['4235 ', '5462 ', '7132 ', '6460 '], 
#     return_tensors="pt"
# ).input_ids  # Batch size 1

def sample(ckpt_dir, step):
    output_dir = os.path.join(ckpt_dir, f'checkpoint-{step}') #f'/hdd3/mohammed/llama_qlora/ckpts/checkpoint-{step}/'
    print(output_dir)
    model = AutoPeftModelForCausalLM.from_pretrained(output_dir, device_map="auto", torch_dtype=torch.bfloat16)
    model = model.merge_and_unload() #returns type transformers.models.llama.modeling_llama.LlamaForCausalLM
    
    outputs = model.generate(input_ids, max_new_tokens=100).cpu()
    out=tokenizer.batch_decode(outputs, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    model = model.cpu()
    return out
s = sample('/hdd3/mohammed/llama_qlora/ckpts', 55)
s

print(s[0])
