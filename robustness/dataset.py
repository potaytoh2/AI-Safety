from config import LABEL_SET, PROMPT_SET
import json

class DataAdvGlue(object):
    def __init__(self,data_path, task):
        self.task = task
        self.data = json.load(open(data_path,'r'))

    def get_data_by_task(self,task):
        self.data_task = self.data[task]
        return self.data_task
        

    def get_content_by_idx(self, idx, task=None):
        if task is None:
            task = self.task
        self.data_task = self.get_data_by_task(task)
        if task == 'sst2':
            content = self.data_task[idx]['sentence']
        elif task == 'qqp':
            content = self.data_task[idx]['question1'] + \
                ' ' + self.data_task[idx]['question2']
        elif task == 'mnli':
            content = self.data_task[idx]['premise'] + \
                ' ' + self.data_task[idx]['hypothesis']
        elif task == 'qnli':
            content = self.data_task[idx]['question'] + \
                ' ' + self.data_task[idx]['sentence']
        elif task == 'rte':
            content = self.data_task[idx]['sentence1'] + \
                ' ' + self.data_task[idx]['sentence2']
        label = self.data_task[idx]['label']
        return content, label

    def get_prompt(self):
        return PROMPT_SET[self.task]

    def get_label(self):
        return LABEL_SET[self.task]

    
class Dataset(object):
    # Data_path is the path that contains the dataset
    # Task contains the dataset task you'll be testing on i.e. sst2, qqp
    def __init__(self, data_name, data_path, task) -> None:
        self.data = data_path
        self.task = task
        self.data_name = data_name
        self.dataclass = self.get_dataclass()

    def get_dataclass(self):
        if self.data_name == 'advglue' or self.data_name == 'advglue_test':
            return DataAdvGlue(self.data, self.task)


    
