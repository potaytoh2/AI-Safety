import json
import pandas as pd

def jsonl_to_excel(jsonl_path, output_excel_path):
    records = []
    with open(jsonl_path, 'r') as f:
        for line in f:
            data = json.loads(line.strip())
            
            # Flatten nested fields for model_output and answer_info
            flat_data = {
                "example_id": data.get("example_id"),
                "category": data.get("category"),
                "question": data.get("question"),
                "context": data.get("context"),
                "label": data.get("label"),
                "ans0_text": data.get("answer_info", {}).get("ans0", ["", ""])[0],
                "ans0_group": data.get("answer_info", {}).get("ans0", ["", ""])[1],
                "ans1_text": data.get("answer_info", {}).get("ans1", ["", ""])[0],
                "ans1_group": data.get("answer_info", {}).get("ans1", ["", ""])[1],
                "ans2_text": data.get("answer_info", {}).get("ans2", ["", ""])[0],
                "ans2_group": data.get("answer_info", {}).get("ans2", ["", ""])[1],
                "prediction": data.get("model_output", {}).get("prediction", ""),
                "prompt_used": data.get("model_output", {}).get("prompt_used", ""),
                "timestamp": data.get("model_output", {}).get("timestamp", "")
            }
            records.append(flat_data)

    # Convert to DataFrame
    df = pd.DataFrame(records)
    df.to_excel(output_excel_path, index=False)
    print(f"Exported to {output_excel_path}")

# Example usage:
jsonl_to_excel("Disability_status_output_200random.jsonl", "Disability_status_output_200random.xlsx")
