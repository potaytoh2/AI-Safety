from config import LABEL_SET, PROMPT_SET, LABEL_TO_ID, DATA_PATH, MODEL_SET
from dataset import Dataset
from model import Model
import argparse
import pandas as pd
from tqdm import tqdm
import numpy as np
import os
import time
import json


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
    parser.add_argument('--self_denoise', action='store_true', default=False)
    parser.add_argument('--mask_rate', type=float, default=0)
    args = parser.parse_args()
    return args

def get_method(idx, task=None):
     data = json.load(open(DATA_PATH[args.dataset],'r'))
     task = data[task]
     return task[idx]['method']
        
def merge_res(args):
    df = pd.DataFrame()
    dataset = args.dataset #advglu
    task = args.task #i.e. sst2
    model_list = MODEL_SET[args.service] # e.g. list of gemini models
    for model in model_list: 
        if args.mask_rate>0:
            if args.self_denoise>0:
                res = pd.read_csv('result/' + dataset + '_' + args.task +
                          '_' + args.service + '_' + model.replace('/', '_') + '_'+ 'maskrate'+ '_' +str(args.mask_rate) + '_self-denoise' + '.csv')
            else:
                res = pd.read_csv('result/' + dataset + '_' + args.task +
                          '_' + args.service + '_' + model.replace('/', '_') + '_'+ 'maskrate'+ '_' +str(args.mask_rate) + '.csv')
        else:
            res = pd.read_csv('result/' + dataset + '_' + args.task +
                          '_' + args.service + '_' + model.replace('/', '_') + '.csv')
        df['idx'] = res['idx']
        df['content'] = res['content']
        df['true_label'] = res['true_label']
        df['pred-'+model.replace('/', '_')] = res['pred_label']
        df['method']=df.index.map(lambda idx: get_method(idx, task=args.task))
    file = f'result/merge_{dataset}_{task}_{args.service}'
    if args.mask_rate>0:
        file += f'_maskrate_{args.mask_rate}'
        if args.self_denoise:
            file += f'_self-denoise'
    file += '.csv'
    df.to_csv(file, index=False)

def compute_metric(pred_label, true_label, task):
    valid_indices = pred_label != -1
    valid_pred_label = pred_label[valid_indices]
    valid_true_label = true_label[valid_indices]

    # Calculate metrics
    num_examples = len(pred_label)  # Total number of examples
    acc = np.mean(pred_label == true_label) * 100.0  # Accuracy considers all predictions
    asr = (1 - np.mean(valid_pred_label == valid_true_label)) * 100.0 if len(valid_pred_label) > 0 else 0.0
    
    return {
        'num_examples': num_examples,
        'acc': acc,
        'asr': asr
    }

def stat(args):
    # Load the appropriate CSV file based on mask_rate
    if args.mask_rate > 0:
        if args.self_denoise:
            df = pd.read_csv(
            f'result/merge_{args.dataset}_{args.task}_{args.service}_maskrate_{str(args.mask_rate)}_self-denoise.csv'
        )
        else:
            df = pd.read_csv(
                f'result/merge_{args.dataset}_{args.task}_{args.service}_maskrate_{str(args.mask_rate)}.csv'
            )
    else:
        df = pd.read_csv(
            f'result/merge_{args.dataset}_{args.task}_{args.service}.csv'
        )

    # Initialize dictionaries to store metrics
    labels = {}  # Contains true labels and predictions
    labels['true_label'] = df['true_label'].to_numpy()
    model_list = MODEL_SET[args.service]

    # Initialize DataFrames for exporting metrics
    combined_metrics_df = pd.DataFrame()  # For combined metrics
    method_metrics_df = pd.DataFrame()    # For metrics by method

    print('For task: ', args.task, ' with maskrate: ', args.mask_rate)

    # Populate predictions for each model
    for model in model_list:
        labels['pred-' + model.replace('/', '_')] = df['pred-' + model.replace('/', '_')].to_numpy()

    # Compute metrics for each method
    grouped = df.groupby('method')
    for method, group in grouped:
        # Extract true labels for the current group
        group_true_labels = group['true_label'].to_numpy()

        for key in labels.keys():
            if key != 'true_label':
                # Filter predictions for the current group
                group_pred_labels = group[key].to_numpy()

                # Map predictions to label IDs
                pred_label = np.array([LABEL_TO_ID[args.task].get(label, -1) for label in group_pred_labels])

                # Compute metrics
                metric_dict = compute_metric(pred_label, group_true_labels, args.task)

                # Add metadata to the metric_dict
                metric_dict['method'] = method
                metric_dict['model'] = key
                metric_dict['mask_rate'] = args.mask_rate
                metric_dict['task'] = args.task

                # Append to method_metrics_df
                method_metrics_df = pd.concat(
                    [method_metrics_df, pd.DataFrame([metric_dict])],
                    ignore_index=True
                )

    # Compute combined metrics for all methods
    for key in labels.keys():
        if key != 'true_label':
            # Map predictions to label IDs
            pred_label = np.array([LABEL_TO_ID[args.task].get(label, -1) for label in labels[key]])

            # Compute metrics
            metric_dict = compute_metric(pred_label, labels['true_label'], args.task)

            # Add metadata to the metric_dict
            metric_dict['model'] = key
            metric_dict['mask_rate'] = args.mask_rate
            metric_dict['task'] = args.task

            # Append to combined_metrics_df
            combined_metrics_df = pd.concat(
                [combined_metrics_df, pd.DataFrame([metric_dict])],
                ignore_index=True
            )

    # Export method-specific metrics to individual CSV files
    for method in method_metrics_df['method'].unique():
        # Filter metrics for the current method
        method_df = method_metrics_df[method_metrics_df['method'] == method]
        file_name = f'result/methods/{method}_metric'
        
        if args.self_denoise:
            file_name +='_self-denoise'

        file_name+='.csv'

        # Append to existing file if it exists
        if os.path.exists(file_name):
            existing_df = pd.read_csv(file_name)
            method_df = pd.concat([existing_df, method_df], ignore_index=True)
        
        # Save to CSV
        method_df.to_csv(file_name, index=False)
        print(f"Exported metrics for method '{method}' to {file_name}")

    # Export combined metrics to a single CSV file
    combined_file_name = f'result/final/combined_metrics_{args.dataset}_{args.task}_{args.service}'

    if args.self_denoise:
        combined_file_name+='_self-denoise'

    combined_file_name+='.csv'
    
    if os.path.exists(combined_file_name):
        existing_df = pd.read_csv(combined_file_name)
        combined_metrics_df = pd.concat([existing_df, combined_metrics_df], ignore_index=True)
    combined_metrics_df.to_csv(combined_file_name, index=False)
    print(f"Combined metrics exported to {combined_file_name}")

def run(args):
    data = Dataset(args.dataset, DATA_PATH[args.dataset], args.task).dataclass
    print('made dataset')
    infer = Model(args.task, args.service, LABEL_SET, MODEL_SET, LABEL_TO_ID, args.model, args.gpu, mask_rate=args.mask_rate, self_denoise = args.self_denoise)
    print("made model")
    data_len = len(data.get_data_by_task(args.task))
    os.makedirs("result", exist_ok=True)
    print('made the dataset and inference')
    args.save_file = 'result/' + args.dataset + '_' + args.task + \
        '_' + args.service + '_' + args.model.replace('/', '_')
    
    if args.mask_rate > 0:
        args.save_file += f'_maskrate_{args.mask_rate}'
        if args.self_denoise:
            args.save_file += f'_self-denoise'

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
        time.sleep(3)
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
