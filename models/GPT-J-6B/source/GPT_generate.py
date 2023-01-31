import os
from happytransformer import GENSettings
from happytransformer import HappyGeneration
import transformers
import torch
from transformers import GPT2LMHeadModel,  GPT2Tokenizer, GPT2Config, GPT2LMHeadModel
from transformers import GPTJForCausalLM

#This code can be used to try GPT-2 with custom inputs
#os.environ["CUDA_VISIBLE_DEVICES"] ="1"
#device = torch.device("cuda")


'''tokenizer = GPT2Tokenizer.from_pretrained(os.getcwd() + "/train/model_save", bos_token='<|startoftext|>', eos_token='<|endoftext|>', pad_token='<|pad|>') #gpt2-medium
configuration = GPT2Config.from_pretrained(os.getcwd() + "/train/model_save", output_hidden_states=False)
model = GPT2LMHeadModel.from_pretrained(os.getcwd() + "/train/model_save", config=configuration)'''

'''
tokenizer = GPT2Tokenizer.from_pretrained("EleutherAI/gpt-j-6B", revision="float16", torch_dtype=torch.float16, low_cpu_mem_usage=True, cache_dir="/srv/nfs-data/mlajko/model_cache")
model = GPT2LMHeadModel.from_pretrained("EleutherAI/gpt-j-6B", revision="float16", torch_dtype=torch.float16, low_cpu_mem_usage=True, cache_dir="/srv/nfs-data/mlajko/model_cache")'''

'''
tokenizer = GPT2Tokenizer.from_pretrained("EleutherAI/gpt-j-6B", revision="float16", torch_dtype=torch.float16, low_cpu_mem_usage=True, cache_dir="/srv/nfs-data/mlajko/model_cache")
model = GPTJForCausalLM.from_pretrained("EleutherAI/gpt-j-6B", revision="float16", torch_dtype=torch.float16, low_cpu_mem_usage=True, cache_dir="/srv/nfs-data/mlajko/model_cache")

model.resize_token_embeddings(len(tokenizer))
model = model.to(device)
'''



def generate(model, tokenizer, input):
    device = torch.device("cuda")
    model.eval()
    prompt = "<|startoftext|>" + input

    generated = torch.tensor(tokenizer.encode(prompt)).unsqueeze(0)
    generated = generated.to(device)

    sample_outputs = model.generate(
        generated,
        do_sample=True,
        top_k=50,
        max_length=1024,
        top_p=0.8,
        num_return_sequences=1
    )

    gen_text = tokenizer.decode(sample_outputs[0], skip_special_tokens=True)
    if gen_text.startswith(input):
        gen_text = gen_text[len(input):]
    output_lines = gen_text.split("\n")
    output_lines = list(filter(None, output_lines))

    return output_lines

#Generating for custom inputs
#print(generate(model, tokenizer, "console.log('Hello, World!');"))
