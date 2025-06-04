import os
import logging
from boto3 import client


logger = logging.getLogger()
#logging.basicConfig(level=logging.DEBUG)
logger.setLevel(level=logging.INFO)

TABLE_NAME = os.environ['TABLE_NAME']

dynamodb_client = client("dynamodb")



def lambda_handler(event, context):

    logger.info(event)

    lake_name = event['Payload']['lakeName']
    segment_index = event['Payload']['segmentIndex']
    metadata = event['Payload']['metadata']
    document = event['Payload']['document']

    item = {
        'lakeName': {'S': lake_name},
        'filePath': {'S': metadata['file_path']},
        'segment_index': {'N': str(segment_index)},
        'document': {'S': document}
    }

    result = dynamodb_client.put_item(
        TableName=TABLE_NAME,
        Item=item
    )

    logger.info(result)
