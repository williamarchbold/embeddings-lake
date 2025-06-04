
# Embeddings Lake

## 🏷 Features

Embeddings Lake is a cost-efficient, open-source VDBMS optimized for the cloud based on the [Vector Lake](https://github.com/msoedov/vector_lake) VDBMS, but with a few critical improvements:

- Significantly better recall that's comparable to commercial VDBMS offerings

- Improved query efficiency for rapid information retrieval

- Migrates both processing and storage to the cloud to circumvent any limitations of local hardware

- Fully serverless cloud application that doesn’t require the user to maintain virtualized machinery


## 📦 Installation

This project uses CDK development with Python.

The `cdk.json` file tells the CDK Toolkit how to execute the app.

This project is set up like a standard Python project.  The initialization
process also creates a virtualenv within this project, stored under the `.venv`
directory.  To create the virtualenv it assumes that there is a `python3`
(or `python` for Windows) executable in your path with access to the `venv`
package. If for any reason the automatic creation of the virtualenv fails,
you can create the virtualenv manually.

To deploy this app, you need an AWS account and environment to deploy to. 

Once you have the account set up clone this repo your local machine. 

```
git clone https://github.com/btreb-ht2/embeddings-lake.git
```

```
cd embeddings-lake
```

Create a virtualenv on MacOS and Linux:

```
$ python3 -m venv .venv
```

After the init process completes and the virtualenv is created, you can use the following
step to activate your virtualenv.

```
$ source .venv/bin/activate
```

If you are a Windows platform, you would activate the virtualenv like this:

```
% .venv\Scripts\activate.bat
```

Once the virtualenv is activated, you can install the required dependencies.

```
$ pip install -r requirements.txt
```

At this point you can now synthesize the CloudFormation template for this code.

```
$ cdk synth
```

Deploy the application to your cloud environment
```
cdk deploy
```


## ⛓️ Quick Start

Use the below python code as an example to test your Embeddings Lake deployment!

```python
import os
import torch
import uuid
import clip
import time
import random
import warnings
import pandas as pd
from PIL import Image
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
warnings.filterwarnings("ignore", category=DeprecationWarning)
random.seed(42)
```

```python
!kaggle datasets download iamsouravbanerjee/animal-image-dataset-90-different-animals/
```

```python
!mkdir Animals-Data
```

```python
!unzip 'animal-image-dataset-90-different-animals.zip' -d 'Animals-Data'
```

```python
IMAGE_DIR = "Animals-Data/animals/animals"
TEST_DIR = "Animals-Test"
```

```python
if not os.path.exists(TEST_DIR):
  for dirpath, dirnames, filenames in os.walk(IMAGE_DIR):
    if filenames:
      class_name = dirpath.split('/')[-1]
      destination_path =  os.path.join(TEST_DIR, class_name)

      os.makedirs(destination_path, exist_ok=True)
      test_image = random.choice(filenames)
      source_path = os.path.join(dirpath, test_image)
      shutil.move(source_path, destination_path)
```

```python
device = "cuda" if torch.cuda.is_available() else "cpu"
# Load Model
model, preprocess = clip.load("ViT-B/32", device=device)
model.eval()
```

```python
train_image_paths = [
    os.path.join(root, file)
    for root, dirs, files in os.walk(IMAGE_DIR)
    for file in files
    if file.endswith((".png", ".jpg", ".jpeg"))
]

test_image_paths = [
    os.path.join(root, file)
    for root, dirs, files in os.walk(TEST_DIR)
    for file in files
    if file.endswith((".png", ".jpg", ".jpeg"))
]

# List of unique ids
train_ids = [str(uuid.uuid4()) for _ in range(len(train_image_paths))]
```

```python
def get_embeddings(image_path):
  image = Image.open(image_path)
  processed_image = preprocess(image).unsqueeze(0)
  with torch.no_grad():
    embeddings = model.encode_image(processed_image.to(device)).float()
  return embeddings.cpu().numpy().flatten().tolist()
```

```python
def get_labels(path):
    return path.split('/')[-2]
```

```python
url_api_lake_add = "https://xxxxxxx.execute-api.us-east-1.amazonaws.com/prod/lake"
url_api_embedding_add = "https://xxxxxxx.execute-api.us-east-1.amazonaws.com/prod/lake/embedding/add"
url_api_embedding_query = "https://xxxxxxxx.execute-api.us-east-1.amazonaws.com/prod/lake/embedding/query"

lake_version = "100"

body_lake_add = {
        "lake_name": f"testLake{lake_version}",
        "lake_dimensions": 512,
        "lake_aprox_shards": 100
      }
```

```python
response_lake_add = requests.put(url=url_api_lake_add, json=body_lake_add)
```

```python
iterator = enumerate(zip(train_image_paths[start_index:], train_ids[start_index:]))

for idx, (image_path, _) in tqdm(iterator,
                            total=len(train_image_paths),
                            initial=start_index,
                            desc=f"Processing images with shards: <{shard}>"):

    label = get_labels(image_path)
    embeddings = get_embeddings(image_path)

    if embeddings is not None and label:

        body_embedding_add = {
            "lake_name": f"testLake{lake_version}",
            "embedding": embeddings,
            "metadata": {"file_path": image_path},
            "document": label,
            "add": True
        }

        response = requests.put(url=url_api_embedding_add, json=body_embedding_add, timeout=240)
```

```python
correct = 0
total_time = 0
incorrect_images = []

for test_image_path in tqdm(
    test_image_paths,
    total=len(test_image_paths),
    desc="Searching from OpenSearch Index"
):
    # search
    test_embedding = get_embeddings(test_image_path)
    start_time = time.time()
    search_body = {
      "query": {
        "knn": {
          "embedding": {
            "vector": test_embedding,
            "k": 1
          }
        }
      }
    }

    response = client.search(index=index_name, body=search_body)
    end_time = time.time()
    total_time += end_time - start_time

    for hit in response['hits']['hits']:
        pred_label = hit['_source']['label']
        true_label = get_labels(test_image_path)
        if pred_label == true_label:
            correct += 1
        else:
            incorrect_images.append((test_image_path, pred_label))
        break

accuracy = correct / len(test_image_paths) if correct else 0.0
print()
print(f"Accuracy: {accuracy}")
print(f"Total Time: {total_time:.2f}")
```