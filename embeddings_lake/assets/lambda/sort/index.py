import os
import logging


logger = logging.getLogger()
logger.setLevel(level=logging.INFO)

BUCKET_NAME = os.environ['BUCKET_NAME']





def lambda_handler(event, context):

    #logger.info(event)
    logger.info(len(event))
    # Event Handler

    all_results = []
    
    for sub_event in event:
        logger.info("sub_event")
        logger.info(sub_event['Payload'])
        for sub_sub_event in sub_event['Payload']:
            logger.info("here02")
            logger.info(sub_sub_event)
            all_results.append(sub_sub_event)

    #logger.info(all_results)

    # sort in ascending order
    sorted_data_desc = sorted(all_results, key=lambda x: x['distance'])

    logger.info("here03")

    logger.info(sorted_data_desc)

    logger.info(sorted_data_desc[0])

    return sorted_data_desc[0]