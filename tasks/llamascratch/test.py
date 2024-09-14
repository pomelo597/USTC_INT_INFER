from transformers import LlamaForCausalLM, AutoTokenizer, GenerationConfig
#from llama import Engine
from llama_copy import Engine
import os
very_long_text = "A path from a point approximately 330 metres east of the most south westerly corner of 17 Batherton Close, Widnes and approximately 208 metres east-south-east of the most southerly corner of Unit 3 Foundry Industrial Estate, Victoria Street, Widnes, proceeding in a generally east-north-easterly direction for approximately 28 metres to a point approximately 202 metres east-south-east of the most south-easterly corner of Unit 4 Foundry Industrial Estate, Victoria Street, and approximately 347 metres east of the most south-easterly corner of 17 Batherton Close, then proceeding in a generally northerly direction for approximately 21 metres to a point approximately 210 metres east of the most south-easterly corner of Unit 5 Foundry Industrial Estate, Victoria Street, and approximately 202 metres east-south-east of the most north-easterly corner of Unit 4 Foundry Industrial Estate, Victoria Street, then proceeding in a generally east-north-east direction for approximately 64 metres to a point approximately 282 metres east-south-east of the most easterly corner of Unit 2 Foundry Industrial Estate, Victoria Street, Widnes and approximately 259 metres east of the most southerly corner of Unit 4 Foundry Industrial Estate, Victoria Street, then proceeding in a generally east-north-east direction for approximately 350 metres to a point approximately 3 metres west-north-west of the most north westerly corner of the boundary fence of the scrap metal yard on the south side of Cornubia Road, Widnes, and approximately 47 metres west-south-west of the stub end of Cornubia Road be diverted to a 3 metre wide path from a point approximately 183 metres east-south-east of the most easterly corner of Unit 5 Foundry Industrial Estate, Victoria Street and approximately 272 metres east of the most north-easterly corner of 26 Ann Street West, Widnes, then proceeding in a generally north easterly direction for approximately 58 metres to a point approximately 216 metres east-south-east of the most easterly corner of Unit 4 Foundry Industrial Estate, Victoria Street and approximately 221 metres east of the most southerly corner of Unit 5 Foundry Industrial Estate, Victoria Street, then proceeding in a generally easterly direction for approximately 45 metres to a point approximately 265 metres east-south-east of the most north-easterly corner of Unit 3 Foundry Industrial Estate, Victoria Street and approximately 265 metres east of the most southerly corner of Unit 5 Foundry Industrial Estate, Victoria Street, then proceeding in a generally east-south-east direction for approximately 102 metres to a point approximately 366 metres east-south-east of the most easterly corner of Unit 3 Foundry Industrial Estate, Victoria Street and approximately 463 metres east of the most north easterly corner of 22 Ann Street West, Widnes, then proceeding in a generally north-north-easterly direction for approximately 19 metres to a point approximately 368 metres east-south-east of the most easterly corner of Unit 3 Foundry Industrial Estate, Victoria Street and approximately 512 metres east of the most south easterly corner of 17 Batherton Close, Widnes then proceeding in a generally east-south, easterly direction for approximately 16 metres to a point approximately 420 metres east-south-east of the most southerly corner of Unit 2 Foundry Industrial Estate, Victoria Street and approximately 533 metres east of the most south-easterly corner of 17 Batherton Close, then proceeding in a generally east-north-easterly direction for approximately 240 metres to a point approximately 606 metres east of the most northerly corner of Unit 4 Foundry Industrial Estate, Victoria Street and approximately 23 metres south of the most south westerly corner of the boundary fencing of the scrap metal yard on the south side of Cornubia Road, Widnes, then proceeding in a generally northern direction for approximately 44 metres to a point approximately 3 metres west-north-west of the most north westerly corner of the boundary fence of the scrap metal yard on the south side of Cornubia Road and approximately 47 metres west-south-west of the stub end of Cornubia Road."

weight_path = os.getenv("HF_MODEL_PATH")
if weight_path is None:
    raise ValueError("HF_MODEL_PATH environment variable is not set")

tokenizer = AutoTokenizer.from_pretrained(weight_path)

#hf_model = LlamaForCausalLM.from_pretrained(weight_path, torch_dtype=torch.float16)   #这样就生成不出来不知道为啥
hf_model = LlamaForCausalLM.from_pretrained(weight_path)

assert type(hf_model) == LlamaForCausalLM
prompt = f"Content: {very_long_text}\n\n Summary:"
prompt_ids = tokenizer(prompt, return_tensors="pt")['input_ids'].view(1, -1)#type:ignore
print("prompt_ids:",prompt_ids)
max_new_tokens = 50    #128不行
gen_config = GenerationConfig(max_new_tokens=max_new_tokens, do_sample=False, return_dict_in_generate=True)

output_ids = hf_model.generate(prompt_ids, gen_config).sequences#type:ignore
print("output_ids:",output_ids)
output_text = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
print("output_text:",output_text)

del tokenizer, hf_model, prompt_ids, gen_config

engine = Engine('/data/jhchen/USTC_INT_INFER/Meta-Llama-3-8B-Instruct')
#llama_output = engine.execute(prompt, max_new_tokens=50, temperature=0.001)[0]
llama_output = engine.execute([prompt], max_new_tokens=50, temperature=0.001)
print("llama_output:",llama_output)
assert output_text == llama_output
