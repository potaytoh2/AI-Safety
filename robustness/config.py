LABEL_SET = {
    'sst2': ['positive', 'negative'],
    'mnli': ['entailment', 'neutral', 'contradiction'],
    'qqp': ['equivalent', 'not_equivalent'],
    'qnli': ['entailment', 'not_entailment'],
    'rte': ['entailment', 'not_entailment'],
}

MODEL_SET = {
    'hug_gen':[
        #"deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        'gemini-1.5-flash',
        'gemini-1.5-flash-8b'
    ],
    'toplace_in_hug_gen':[
        'mistralai/Ministral-8B-Instruct-2410',
    ]
}

PROMPT_SET = {
    'sst2': [
        'Is the following sentence positive or negative? Only answer me with "positive" or "negative", just one word. ',
    ],
    'qqp': [
        'Are the following two questions equivalent or not? Only answer me with "equivalent" or "not_equivalent", just one word. ',
    ],
    'mnli': [
        'Are the following two sentences entailment, neutral or contradiction? Only answer me with "entailment", "neutral" or "contradiction", just one word. ',
    ],
    'qnli': [
        'Are the following question and sentence entailment or not_entailment? Only answer me with "entailment" or "not_entailment", just one word. ',
    ],
    'rte': [
        'Are the following two sentences entailment or not_entailment? Only answer me with "entailment" or "not_entailment", just one word. ',
    ],
}

LABEL_TO_ID = {
    'sst2': {'negative': 0, 'positive': 1, 'neutral': 2},
    'mnli': {'entailment': 0, 'neutral': 1, 'contradiction': 2},
    'qqp': {'equivalent': 1, 'not_equivalent': 0},
    'qnli': {'entailment': 0, 'not_entailment': 1},
    'rte': {'entailment': 0, 'not_entailment': 1},
}

ID_TO_LABEL = {
    'sst2': {0: 'negative', 1: 'positive', 2: 'neutral'},
    'mnli': {0: 'entailment', 1: 'neutral', 2: 'contradiction'},
    'qqp': {1: 'equivalent', 0: 'not_equivalent'},
    'qnli': {0: 'entailment', 1: 'not_entailment'},
    'rte': {0: 'entailment', 1: 'not_entailment'},
}

#Change the mapping when you move dev.json
DATA_PATH = {
    'advglue': './dev.json',
    'advglue_test': './sample.json'
}
