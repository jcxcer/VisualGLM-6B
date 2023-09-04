from transformers import AutoTokenizer, AutoModel
import torch
import sys
import requests
import hashlib
import os
import json

HISTORY_PATH = "/var/pyproj/VisualGLM-6B/history"


def saveHistory(conversation_key, history):
    with open(f"{HISTORY_PATH}/{conversation_key}.txt", "w", encoding="utf-8") as f:
        f.write(json.dumps(history, ensure_ascii=False))


def loadHistory(conversation_key):
    history = []
    if os.path.exists(f"/{HISTORY_PATH}/{conversation_key}.txt"):
        with open(f"{HISTORY_PATH}/{conversation_key}.txt", "r", encoding="utf-8") as f:
            history = f.read()
            history = json.loads(history)
    return history


def main():
    query_text = ''
    query_img = ''
    query_key = ''
    query_file_name = ''

    query_file = sys.argv[1]
    if os.path.exists(f"{HISTORY_PATH}/tmp/" + query_file):
        with open(f"{HISTORY_PATH}/tmp/" + query_file, "r", encoding="utf-8") as f:
            query = f.read()
            query = json.loads(query)

            query_text = query["text"]
            query_img = query["img"]
            if 'key' in query:
                query_key = query["key"]
            if 'file_name' in query:
                query_file_name = query["file_name"]

    if query_text == '' or query_img == '':
        print('output:query or img load error')
        return

    history = []
    if query_key:
        history = loadHistory(query_key)

    file_hash = hashlib.md5(query_img.encode('utf-8')).hexdigest()
    if query_file_name == '':
        query_file_name = os.path.basename(query_img)

    filepath = "/var/pyproj/VisualGLM-6B/history/" + file_hash + "/"

    if not os.path.exists(filepath):
        os.mkdir(filepath)

    filepath += query_file_name
    if not os.path.exists(filepath):
        file_content = requests.get(query_img)
        open(filepath, 'wb').write(file_content.content)

    tokenizer = AutoTokenizer.from_pretrained("/var/pyproj/VisualGLM-6B/models/visualglm-6b", trust_remote_code=True)
    model = AutoModel.from_pretrained("/var/pyproj/VisualGLM-6B/models/visualglm-6b",
                                      trust_remote_code=True).half().cuda()
    model = model.eval()

    with torch.no_grad():
        response, history = model.chat(tokenizer, filepath, query, history=history)

    print("output: ", response)

    if query_key:
        saveHistory(query_key, history)


if __name__ == "__main__":
    main()
