import functions_framework
from anthropic import AnthropicVertex
from chardet import detect
import zipfile
import tempfile
import json
import os

import base64
import vertexai
from vertexai.generative_models import GenerativeModel, Part, FinishReason
import vertexai.preview.generative_models as generative_models

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = './credential.json'

@functions_framework.http
def main(request):
    try:
        repo = {'files': []}

        def rec(path):
            if os.path.isdir(path):
                files = os.listdir(path)
                for f in files:
                    rec(path + "/" + f)
            else:
                # if path.split('.')[-1] == '.py':
                with open(path, 'rb') as bf:
                    binary_data = bf.read()
                    encode_data = detect(binary_data)
                if encode_data['encoding'] is not None:
                    with open(path, encoding=encode_data['encoding']) as f:
                        repo['files'].append({'path': '/'+'/'.join(path.split('/')[3:]), 'content': f.read()})

        f = request.files['file']
        with tempfile.TemporaryDirectory() as td:
            with zipfile.ZipFile(f) as zf:
                zf.extractall(path=td)
                headers = {
                    'Access-Control-Allow-Origin': '*'
                }
                rec(td)
                system_prompt = request.form.get('system_prompt')
                prompt = request.form.get('prompt') # プロンプト
                model_name = request.form.get('model_name') # モデル名
                repo_text = json.dumps(repo, indent=4, ensure_ascii=False)  # JSON形式の全ソースコード
                response = generate(repo_text, system_prompt, prompt, model_name)
                return (json.dumps({"response": response}), 200, headers)
    except Exception as e:
        print(e)
        headers = {
            'Access-Control-Allow-Origin': '*'
        }
        return (json.dumps({"status": 500, "message": "Internal Server Error"}), 500, headers)

def generate(repo_text, system_prompt, prompt, model_name):
    if model_name == "gemini-1.5-pro-001":
        vertexai.init(project="XXXXX", location="asia-northeast1")
        model = GenerativeModel(model_name="gemini-1.5-pro-001", system_instruction=[system_prompt])
        response = model.generate_content(f"""#指示:
{prompt}

#ソースコード:
{repo_text}""",
            generation_config={
                "max_output_tokens": 8192,
                "top_p": 0.95,
            },
            safety_settings={
                generative_models.HarmCategory.HARM_CATEGORY_HATE_SPEECH: generative_models.HarmBlockThreshold.BLOCK_ONLY_HIGH,
                generative_models.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: generative_models.HarmBlockThreshold.BLOCK_ONLY_HIGH,
                generative_models.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: generative_models.HarmBlockThreshold.BLOCK_ONLY_HIGH,
                generative_models.HarmCategory.HARM_CATEGORY_HARASSMENT: generative_models.HarmBlockThreshold.BLOCK_ONLY_HIGH,
            },
        )

        return response.text
    elif model_name == "claude-3-5-sonnet@20240620":
        client = AnthropicVertex(region="europe-west1", project_id="XXXXX")

        message = client.messages.create(
        max_tokens=4096,
        messages=[
            {
            "role": "user",
            "content": f"""#指示:
{prompt}

#ソースコード:
{repo_text}""",
            }
        ],
        model="claude-3-5-sonnet@20240620",
        )
        
        return message.content[0].text