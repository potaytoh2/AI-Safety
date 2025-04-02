import torch
from dotenv import load_dotenv
from google import genai
from google.genai import types
import os
import time
from collections import defaultdict
import random
import math

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=GEMINI_API_KEY)

class Model(object):

    def __init__(self,
                 task,
                 service,
                 label_set,
                 model_set,
                 label_to_id,
                 model=None,
                 gpu=-1,
                 mask_rate=0):
        
        self.task = task
        self.service = service
        self.model = model
        self.label_set = label_set
        self.model_set = model_set
        self.label_to_id = label_to_id
        self.gpu = gpu
        self.mask_rate = mask_rate

        if self.service == 'hug_gen' and "deepseek" in self.model:
            print("initializing")
            from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
            self.tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-7B")
            model = AutoModelForCausalLM.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-7B", device_map="auto", load_in_8bit=True)
            self.pipe = pipeline("text-generation", model=model, tokenizer=self.tokenizer, device_map='auto')
            print("initialized")
        print("using model:", self.model)
    #TODO: need to configure this for the other models that we'll be using.        
    def predict(self, sentence, prompt = None):
        if self.service == 'hug_gen':
            if 1 >= self.mask_rate > 0:
                res = self.pred_with_mask(prompt, sentence)
            else:
                res = self.pred_by_generation(prompt, sentence, self.label_set[self.task])
            return res

        return None

    def pred_with_mask(self, prompt, sentence):
        sen_list = sentence.split()
        sen_len = len(sen_list)
        rem_len = math.floor(sen_len - self.mask_rate * sen_len)
        print("This is remaining length",rem_len)
        n = sen_len - rem_len

        def classifier_res(x, hx, kx, n):
            B = []
            
            for _ in range(n):
                H = random.sample(range(hx), kx)
                B.append(H)
            
            counts = defaultdict(int)
            for H in B:
                tmp_sentence = x.copy()
                for i in range(hx):
                    if i not in H:
                        tmp_sentence[i] = '[MASK]'

                new_sentence = " ".join(tmp_sentence)
                print(new_sentence)
                # Get prediction from model selected
                c = self.pred_by_generation(prompt, new_sentence, self.label_set[self.task])
                counts[c] += 1

            # Return counts
            return counts


        counts = classifier_res(sen_list, sen_len, rem_len, n)
        max_count = -1
        res = None
        for c in counts:
            if counts[c]>max_count:
                max_count = counts[c]
                res = c
        
        return res


    def pred_by_generation(self, prompt, sentence, label_set):
        
        def process_label(pred_label, label_set):
            label_set.sort(key=len, reverse=True)
            for item in label_set:
                if item.lower() in pred_label.lower():
                    return item.lower()
            return pred_label
        
        def find_label(output):
            labels = self.label_set[self.task]
            labels.sort(key=len,reverse=True)
            max_len = len(labels[0])+20
            last_output = output[-max_len:].lower()
            for i in labels:
                if i.lower() in last_output:
                    return i
            return output

        out = 'error!'
        input_text = prompt + sentence + ' Answer: '

        if "deepseek" in self.model:
            inputs_ids = self.tokenizer(input_text)['input_ids']
            out = self.pipe(input_text, max_length=5000)
            out = out[0]['generated_text']
            out=out.split(':')[-1].strip()
        elif "gemini" in self.model:
            out = client.models.generate_content(
                model=self.model,
                contents=[input_text],
                config=types.GenerateContentConfig(
                    max_output_tokens=20,
                    temperature=0.1,
                    top_k=1
                )
            ).text
            print("This is output",out)
            time.sleep(5)

        if 'deepseek' in self.model.lower():
            end_index = out.find('</think>')
            if end_index == -1:
                out_processed = find_label(out)
            else:
                string_aft_think = out[end_index+len('</think>'):]
                out_processed = string_aft_think.replace('\n','')
        else:
            out_processed = process_label(out, label_set)
        
        return out_processed
