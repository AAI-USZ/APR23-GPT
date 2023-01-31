import os
import transformers
import torch
import os
import openai
import time
import urllib, json

#This code can be used to try GPT-2 with custom inputs
#os.environ["CUDA_VISIBLE_DEVICES"] ="1"
#device = torch.device("cuda")




def generate(model, tokenizer, input):

    #prompt = "<|startoftext|>" + input
    prompt = input


    openai.api_key = "OpenAI_Key"
    #print(input)

    rate_limit_per_minute = 20
    delay = 60.0 / rate_limit_per_minute * 1.1
    #print("Delay: ", delay)
    time.sleep(delay)
    
    while (True):
        try:
            response = openai.Completion.create(
            model="code-davinci-002",
            prompt=input,
            temperature=0.6,
            max_tokens=256,
            top_p=0.8,
            frequency_penalty=0.0,
            presence_penalty=0.0
            )
            break
        except:
            print("OpenAI Error, new request 10 seconds later")
            time.sleep(10)
    
    #print(response)
    #print(response["choices"][0]["text"])
    print(len(response["choices"][0]["text"]))
    ##time.sleep(1000)

    
    output_lines = response["choices"][0]["text"].split("\n")
    output_lines = list(filter(None, output_lines))
    return output_lines

#Generating for custom inputs
#print(generate(model, tokenizer, "console.log('Hello, World!');"))
