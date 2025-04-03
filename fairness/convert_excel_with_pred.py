import json
import pandas as pd
import re

# Load JSONL file
jsonl_path = "/common/home/users/y/ythuang.2022/AI-Safety/fairness/output/Sexual_orientation_output.jsonl"
records = []

with open(jsonl_path, 'r') as f:
    for line in f:
        data = json.loads(line.strip())
        model_output = data.get("model_output", {}).get("prediction", "")
        
        # Extract answer text from answer_info
        ans0_text = data.get("answer_info", {}).get("ans0", ["", ""])[0]
        ans1_text = data.get("answer_info", {}).get("ans1", ["", ""])[0]
        ans2_text = data.get("answer_info", {}).get("ans2", ["", ""])[0]

        # Normalize prediction and answer options
        def clean(text):
            return text.lower().strip().strip('.').replace("\n", " ")

        prediction_clean = clean(model_output)
        ans_map = {
            "ans0": clean(ans0_text),
            "ans1": clean(ans1_text),
            "ans2": clean(ans2_text)
        }

        matched_prediction = "unmatched"
        for key, value in ans_map.items():
            if value in prediction_clean:
                matched_prediction = key
                break

        # Add everything to the record
        data['ans0_text'] = ans0_text
        data['ans1_text'] = ans1_text
        data['ans2_text'] = ans2_text
        data['prediction_extracted'] = matched_prediction
        records.append(data)

# Convert to DataFrame and export to Excel
df = pd.DataFrame(records)
output_excel_path = "/common/home/users/y/ythuang.2022/AI-Safety/fairness/Sexual_orientation_with_predictions.xlsx"
df.to_excel(output_excel_path, index=False)

output_excel_path