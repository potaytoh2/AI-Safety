from config import LABEL_SET, PROMPT_SET, LABEL_TO_ID, DATA_PATH, MODEL_SET
from dataset import Dataset
from model import Model
import argparse
import pandas as pd
from tqdm import tqdm
import numpy as np
import os

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str,
                        default='deepseek-ai/DeepSeek-R1-Distill-Qwen-7B')
    parser.add_argument(
        '--data_path', type=str, default='data/advglue/dev.json')
    parser.add_argument('--task', type=str, default='mnli')
    parser.add_argument('--gpu', type=int, default=-1)
    parser.add_argument('--save_file', type=str, default='result/result.csv')
    parser.add_argument('--service', type=str, default='chat')
    parser.add_argument('--dataset', type=str, default='advglue')
    parser.add_argument('--eval', action='store_true', default=False)
    parser.add_argument('--mask_rate', type=float, default=0)
    args = parser.parse_args()
    return args

def merge_res(args):
    df = pd.DataFrame()
    dataset = args.dataset #advglu
    task = args.task #i.e. sst2
    model_list = MODEL_SET[args.service] # e.g. list of gemini models
    for model in model_list: 
        res = pd.read_csv('result/' + dataset + '_' + args.task +
                          '_' + args.service + '_' + model.replace('/', '_') + '.csv')
        df['idx'] = res['idx']
        df['content'] = res['content']
        df['true_label'] = res['true_label']
        df['pred-'+model.replace('/', '_')] = res['pred_label']
    df.to_csv(
        f'result/merge_{dataset}_{task}_{args.service}.csv', index=False)


def compute_metric(pred_label, true_label, task):
    return {'num_examples': len(pred_label), 
            'acc': np.mean(pred_label == true_label) * 100.0, 
            'asr': 100.0 - np.mean(pred_label == true_label) * 100.0}


def stat(args):
    df = pd.read_csv(
        f'result/merge_{args.dataset}_{args.task}_{args.service}.csv')
    labels = {}
    labels['true_label'] = df['true_label'].to_numpy()
    model_list = MODEL_SET[args.service]
    for model in model_list:
        labels['pred-'+model.replace('/', '_')] = df['pred-' +
                                                     model.replace('/', '_')].to_numpy()
    for key in labels.keys():
        if key != 'true_label':
            pred_label = []
            for label in labels[key]:
                pred_label.append(LABEL_TO_ID[args.task].get([label],-1))
            pred_label = np.array(pred_label)
            # acc = np.mean(labels[key] == true_label)
            metric_dict = compute_metric(pred_label, labels['true_label'], args.task)
            
            metric_string = ', '.join(
                ['{:s}:{:.2f}'.format(k, v) for k, v in metric_dict.items()])
            print("{:s} - {:s}".format(key, metric_string))


def run(args):
    data = Dataset(args.dataset, DATA_PATH[args.dataset], args.task).dataclass
    print('made dataset')
    infer = Model(args.task, args.service, LABEL_SET, MODEL_SET, LABEL_TO_ID, args.model, args.gpu, mask_rate=args.mask_rate)
    print("made model")
    data_len = len(data.get_data_by_task(args.task))
    os.makedirs("result", exist_ok=True)
    print('made the dataset and inference')
    args.save_file = 'result/' + args.dataset + '_' + args.task + \
        '_' + args.service + '_' + args.model.replace('/', '_')
    
    if args.mask_rate > 0:
        args.save_file += f'_maskrate_{args.mask_rate}'

    args.save_file += '.csv'
    lst = []
    for idx in tqdm(range(data_len)):
        res_dict = {}
        content, label = data.get_content_by_idx(idx, args.task)
        pred_label = infer.predict(
            content, prompt=PROMPT_SET[args.task][-1])
        res_dict['idx'] = idx
        res_dict['content'] = content
        res_dict['true_label'] = label
        res_dict['pred_label'] = pred_label
        lst.append(res_dict)

    pd.DataFrame(lst).to_csv(args.save_file, index=False)
    print(f"✅ CSV file saved successfully: {args.save_file}")
    print(f"done predicting for: {args.task}")

if __name__ == '__main__':
    args = get_args()
    if not args.eval:
        print("running args")
        run(args)
        print("done running args")
    else:
        merge_res(args)
        stat(args)
