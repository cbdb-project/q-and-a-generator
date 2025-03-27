import pandas as pd
import requests
import json

output_language = "English"

# Define the prompt template without f-string
prompt = f'''You will be given a paragraph of text. The paragraph may be written in any language. Your task is to extract **5 fact-based, binary (True/False) question and answer pairs** based strictly on the content of the paragraph. Follow these rules:

- Each question must be answerable with **True** or **False** based on the paragraph.  
- The **language of the questions** should be **{output_language}**, regardless of the input paragraph's language.  
- The **answers** must be the Boolean values `true` or `false` (not strings or capitalized).  
- Each item must also include a **reference**, which is a direct quote (in the original language) from the input paragraph that supports the answer.  
- Return the result as a **JSON array** of 5 objects.  
- Each object must contain exactly **three** fields: `question`, `answer`, and `reference`.  
- Do **not** include any extra explanation, text, or formatting outside the JSON structure.

**Expected Output:**
[
    {{
        "question": "...",
        "answer": true,
        "reference": "..."
    }},
    {{
        "question": "...",
        "answer": false,
        "reference": "..."
    }},
    {{
        "question": "...",
        "answer": true,
        "reference": "..."
    }},
    {{
        "question": "...",
        "answer": false,
        "reference": "..."
    }},
    {{
        "question": "...",
        "answer": true,
        "reference": "..."
    }}
]'''

input_df = pd.read_csv('input.csv')

with open('api_token.txt', 'r') as f:
    api_token = f.read().strip()

def chatgpt_api_call(prompt):
    headers = {
        "Authorization": f"Bearer {api_token}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": "gpt-4o",
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.5
    }
    
    response = requests.post(
        "https://go.apis.huit.harvard.edu/ais-openai-direct-limited-schools/v1/chat/completions",
        headers=headers,
        json=payload
    )
    
    if response.status_code == 200:
        response_json = response.json()
        return response_json["choices"][0]["message"]["content"]
    else:
        print(f"Error: {response.status_code}")
        print(response.text)
        return None

output_list = []
for index, row in input_df.iterrows():
    print(f"Working on document ID: {row['id']}")
    document = row['document']
    document_id = row['id']
    prompt_with_document = prompt + document
    # print(prompt_with_document)
    response = chatgpt_api_call(prompt_with_document)
    if response.startswith("```json"):
        response = response[8:-3]
    
    max_retries = 2
    retries = 0
    valid_json = False
    
    while retries < max_retries and not valid_json:
        try:
            json_response = json.loads(response)
            valid_json = True
        except json.JSONDecodeError:
            print(f"Invalid JSON received, retrying ({retries + 1}/{max_retries})...")
            retries += 1
            response = chatgpt_api_call(prompt_with_document)
            if response.startswith("```json"):
                response = response[8:-3]
    
    if valid_json:
        for item in json_response:
            try:
                question = item["question"]
                answer = item["answer"]
                reference = item["reference"]
                output_list.append([document_id, question, answer, reference])
            except KeyError as e:
                print(f"KeyError: {e} in document ID {document_id}")
                print(f"JSON Response: {json_response}")
        # print(json_response)
        output_df = pd.DataFrame(output_list, columns=["document_id", "question", "answer", "reference"])
        output_df.to_csv('output.csv', index=False)
    else:
        print("Could not get valid JSON after retries")