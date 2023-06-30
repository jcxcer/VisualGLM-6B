from transformers import AutoTokenizer, AutoModel
import torch
import sys
import requests
import hashlib
import os

tokenizer = AutoTokenizer.from_pretrained("/var/pyproj/VisualGLM-6B/models/visualglm-6b", trust_remote_code=True)
model = AutoModel.from_pretrained("/var/pyproj/VisualGLM-6B/models/visualglm-6b", trust_remote_code=True).half().cuda()
model = model.eval()


def main():
    history = []

    file_url = sys.argv[2]
    file_name = sys.argv[3]
    file_content = requests.get(file_url)
    file_hash = hashlib.md5(file_content.content).hexdigest()
    if file_name == '':
        file_name = os.path.basename(file_url)

    filepath = "/var/pyproj/VisualGLM-6B/history/" + file_hash + "/"
    if not os.path.exists(filepath):
        os.mkdir(filepath)

    filepath += file_name
    if not os.path.exists(filepath):
        open(filepath, 'wb').write(file_content.content)

    image_path = filepath
    query = sys.argv[1]

    with torch.no_grad():
        response, history = model.chat(tokenizer, image_path, query, history=history)

    print("output: ", response)


if __name__ == "__main__":
    main()
