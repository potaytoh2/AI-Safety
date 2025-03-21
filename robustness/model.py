import torch

class Model(object):

    def __init__(self,
                 task,
                 service,
                 label_set,
                 model_set,
                 label_to_id,
                 model=None,
                 gpu=-1):
        self.task = task
        self.service = service
        self.model = model
        self.label_set = label_set
        self.model_set = model_set
        self.label_to_id = label_to_id
        self.gpu = gpu

        if self.service == 'hug_gen' and "deepseek" in self.model:
            print("initializing")
            from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
            self.tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-7B")
            model = AutoModelForCausalLM.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-7B", device_map="auto", load_in_8bit=True)
            self.pipe = pipeline("text-generation", model=model, tokenizer=self.tokenizer, device_map='auto')
            print("initialized")

    #TODO: need to configure this for the other models that we'll be using.        
    def predict(self, sentence, prompt = None):
        if self.service == 'hug_gen' and "deepseek" in self.model:
            res = self.pred_by_generation(prompt, self.model, sentence, self.label_set[self.task])
            return res
        return None
    
    def pred_by_generation(self, prompt, model, sentence, label_set):
        def process_label(pred_label, label_set):
            for item in label_set:
                if item.lower() in pred_label.lower():
                    return item
            return pred_label
        
        out = 'error!'
        input_text = prompt + sentence + ' Answer: '

        if self.service == 'hug_gen' and "deepseek" in self.model:
            inputs_ids = self.tokenizer(input_text)['input_ids']
            out = self.pipe(input_text, max_length=5000)
            out = out[0]['generated_text']
            out=out.split(':')[-1].strip()
        
        if 'deepseek' in model.lower():
            # end_index = out.find('</think>')
            # string_aft_think = out[end_index+len('</think>'):]
            # out_processed = string_aft_think.replace('\n','')
            out_processed = out
        else:
            out_processed = process_label(out, label_set)
        
        return out_processed
