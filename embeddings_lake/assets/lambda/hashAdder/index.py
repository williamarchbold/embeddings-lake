from numpy import array as np_array, dot as np_dot, random as np_random
from boto3 import resource as boto3_resource
from boto3 import client as boto3_client
import os
import json
import logging

logger = logging.getLogger()
logger.setLevel(level=logging.ERROR)


s3_resource = boto3_resource("s3")
sqs_client = boto3_client("sqs")


BUCKET_NAME = os.environ['BUCKET_NAME']
QUEUE_URL = os.environ['QUEUE_URL']


def lsh(vector, hyperplanes):
    """Hashes a vector using all hyperplanes.

    Returns a string of 0s and 1s.
    """
    return int(
        "".join(
            [
                "1" if np_dot(hyperplane, vector) > 0 else "0"
                for hyperplane in hyperplanes
            ]
        ),
        base=2,
    )    

def vector_router(vector: np_array, hyperplanes) -> int:
    if isinstance(vector, list):
        vector = np_array(vector)
    closest_index = lsh(vector, hyperplanes)
    return closest_index

def lambda_handler(event, context):

    logger.info(event)

    lake_name = event['lake_name']
    embedding = event['embedding']
    add_embedding = event['add']


    object = s3_resource.Object(BUCKET_NAME, f"{lake_name}/lake_config.json")
    object_contents = object.get()["Body"].read().decode("utf-8")
    lake_config = json.loads(object_contents)
    logger.info(f"Lake config - {lake_config}")
    
    hyperplanes = lake_config["lake_hyperplanes"]
    num_shards = lake_config["lake_shards"]
    shard_index = vector_router(np_array(embedding), hyperplanes)

    result = { 
        'lake_name': lake_name,
        'embedding': embedding,
        'segment_index': shard_index,
        'num_shards': num_shards,
        'add': add_embedding,
    }
    
    if add_embedding:
        document = event['document']
        metadata = event['metadata']
        result['document'] = document
        result['metadata'] = metadata

        messsage_body = json.dumps(result)

        sqs_client.send_message(
            QueueUrl=QUEUE_URL,
            MessageBody=messsage_body,
            MessageGroupId=lake_name
        )

        return "message sent"

    else:
        return "failure"
