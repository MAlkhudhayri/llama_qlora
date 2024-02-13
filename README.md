Based on the QLoRA repo

Setup: `conda env create -f environment.yml`

Bug modification: make line 818 to `hidden_states = outputs[0]**.to(torch.bfloat16)**` 
in transformers/models/llama/modeling_llama.py to avoid datatype issues

