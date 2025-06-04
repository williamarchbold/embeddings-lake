import aws_cdk as core
import aws_cdk.assertions as assertions

from embeddings_lake.embeddings_lake_stack import EmbeddingsLakeStack

# example tests. To run these tests, uncomment this file along with the example
# resource in embeddings_lake/embeddings_lake_stack.py
def test_sqs_queue_created():
    app = core.App()
    stack = EmbeddingsLakeStack(app, "embeddings-lake")
    template = assertions.Template.from_stack(stack)

#     template.has_resource_properties("AWS::SQS::Queue", {
#         "VisibilityTimeout": 300
#     })
