import websocket #NOTE: websocket-client (https://github.com/websocket-client/websocket-client)
import uuid
import json
import urllib.request
import urllib.parse
import numpy as np
from PIL import Image
import io

MAX_SEED=np.iinfo(np.int32).max

# Set to ComfyUI running address and port
server_address = "127.0.0.1:6621"
client_id = str(uuid.uuid4())

def queue_prompt(prompt):
    p = {"prompt": prompt, "client_id": client_id}
    data = json.dumps(p).encode('utf-8')
    req =  urllib.request.Request("http://{}/prompt".format(server_address), data=data)
    return json.loads(urllib.request.urlopen(req).read())

def get_image(filename, subfolder, folder_type):
    data = {"filename": filename, "subfolder": subfolder, "type": folder_type}
    url_values = urllib.parse.urlencode(data)
    with urllib.request.urlopen("http://{}/view?{}".format(server_address, url_values)) as response:
        return response.read()

def get_history(prompt_id):
    with urllib.request.urlopen("http://{}/history/{}".format(server_address, prompt_id)) as response:
        return json.loads(response.read())

def get_images(ws, prompt):
    prompt_id = queue_prompt(prompt)['prompt_id']
    output_images = {}
    while True:
        out = ws.recv()
        if isinstance(out, str):
            message = json.loads(out)
            if message['type'] == 'executing':
                data = message['data']
                if data['node'] is None and data['prompt_id'] == prompt_id:
                    break #Execution is done
        else:
            continue #previews are binary data

    history = get_history(prompt_id)[prompt_id]
    for o in history['outputs']:
        for node_id in history['outputs']:
            node_output = history['outputs'][node_id]
            if 'images' in node_output:
                images_output = []
                for image in node_output['images']:
                    image_data = get_image(image['filename'], image['subfolder'], image['type'])
                    images_output.append(image_data)
            output_images[node_id] = images_output

    return output_images


with open("Coin3D_condition_workflow_api.json", 'r') as f:
    prompt = json.load(f)

# Load Image Node, set to your condition image path.
prompt["189"]['inputs']['image'] = "path/to/your/condition.png" # /mnt/projects/Coin3D/example/teddybear/condition.png

# Text Encode Node, set as your text prompt. Positive prompt
prompt["16"]['inputs']['text'] = "a lovely teddy bear" 

# Negative prompt
# prompt["17"]['inputs']['text'] = "ugly, global light"

# Depth-condition: ControlNetApplyAdvanced Node, set as your text prompt.
prompt["176"]['inputs']['strength'] = 0.83
prompt["176"]['inputs']['end_percent'] = 0.791
prompt["176"]['inputs']['start_percent'] = 0.0

# Softedge-condition: ControlNetApplyAdvanced Node, set as your text prompt.
prompt["13"]['inputs']['strength'] = 0.87
prompt["13"]['inputs']['end_percent'] = 0.665
prompt["13"]['inputs']['start_percent'] = 0.0

prompt["3"]['inputs']['seed'] = np.random.randint(0, MAX_SEED)

ws = websocket.WebSocket()
ws.connect("ws://{}/ws?clientId={}".format(server_address, client_id))
images = get_images(ws, prompt)['9'] # '9' generated images.

for idx, image_data in enumerate(images):
    image = Image.open(io.BytesIO(image_data))
    image.save(f"{idx}.png")