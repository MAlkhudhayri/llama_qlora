Based on the QLoRA repo

Bug modification: make line 818 to `hidden_states = outputs[0]**.to(torch.bfloat16)**` 
in transformers/models/llama/modeling_llama.py to avoid datatype issues

