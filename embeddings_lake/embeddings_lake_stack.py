from aws_cdk import (
    Duration,
    Stack,
    aws_s3 as s3,
    aws_lambda as lambda_,
    aws_stepfunctions_tasks as tasks,
    aws_stepfunctions as sfn,
    aws_apigateway as apigateway,
    aws_dynamodb as dynamodb,
    aws_iam as iam,
    aws_logs as logs,
    aws_sqs as sqs,
    aws_lambda_event_sources as lambda_event_sources,
    BundlingOptions
)
from constructs import Construct

class EmbeddingsLakeStack(Stack):

    def __init__(self, scope: Construct, construct_id: str, **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)


        lambda_endpoint = "lambda.amazonaws.com"

        bucket_segments = s3.Bucket(
            self, 
            "BucketSegments",
            encryption=s3.BucketEncryption(s3.BucketEncryption.S3_MANAGED),
            object_ownership=s3.ObjectOwnership.BUCKET_OWNER_ENFORCED,
            block_public_access=s3.BlockPublicAccess.BLOCK_ALL,
        )

        table_embeddings = dynamodb.TableV2(
            self,
            id="TableEmbeddings",
            partition_key=dynamodb.Attribute(
                name="lakeName",
                type=dynamodb.AttributeType.STRING
            ),
            sort_key=dynamodb.Attribute(
                name="filePath",
                type=dynamodb.AttributeType.STRING
            )
        )

        queue_embeddings_add = sqs.Queue(
            self,
            "QueueEmbeddingsAdd",
            visibility_timeout=Duration.seconds(600),
            content_based_deduplication=True,
            fifo=True,
        )

        queue_dead_letter_embeddings_add = sqs.Queue(
            self,
            "QueueDeadLetterEmbeddingsAdd",
            visibility_timeout=Duration.seconds(600),
            fifo=False,
        )


        # layer_bundling_command = (
        #     "pip install -r requirements.txt "
        #     "-t /asset-output/python && "
        #     "find /asset-output/python -type d -name '__pycache__' -exec rm -rf {} + && "
        #     "cp -au . /asset-output/python"
        # )

        # lambda_layer_pydantic = lambda_.LayerVersion(
        #     self,
        #     "numpyLambdaLayer",
        #     compatible_runtimes=[lambda_.Runtime.PYTHON_3_10],
        #     code=lambda_.Code.from_asset(
        #         "embeddings_lake/assets/lambda/layers/pydantic",
        #         bundling=BundlingOptions(
        #             image=lambda_.Runtime.PYTHON_3_10.bundling_image,
        #             command=[
        #                 "bash",
        #                 "-c",
        #                 layer_bundling_command,
        #             ]
        #         )
        #     )
        # )

        # https://www.youtube.com/watch?v=jyuZDkiHe2Q
        lambda_layer_pydantic = lambda_.LayerVersion(
            self,
            "pydanticLambdaLayer",
            code=lambda_.Code.from_asset("embeddings_lake/assets/lambda/layers/pydantic/lambda-layer-pydantic.zip"),
        )

        lambda_layer_pandas = lambda_.LayerVersion.from_layer_version_arn(
            self,
            "pandasLambdaLayer",
            layer_version_arn="arn:aws:lambda:us-east-1:336392948345:layer:AWSSDKPandas-Python310:23"
        )

        function = lambda_.Function(
            self,
            "EmbeddingsFunction",
            runtime=lambda_.Runtime.PYTHON_3_10,
            handler="index.lambda_handler",
            code=lambda_.Code.from_asset("embeddings_lake/assets/lambda/embedder"),
            layers=[lambda_layer_pandas, lambda_layer_pydantic]
        )

        policy_lake_instantiate = iam.ManagedPolicy(
            self,
            "PolicyLambdaLakeInstantiation",
            managed_policy_name="EmbeddingsLake_LambdaLakeInstation",
            document=iam.PolicyDocument(
                statements=[
                    iam.PolicyStatement(
                        effect=iam.Effect.ALLOW,
                        actions=[
                            "s3:PutObject"
                        ],
                        resources=[
                            bucket_segments.bucket_arn,
                            bucket_segments.arn_for_objects("*"),
                        ],
                    )
                ]
            )
        )

        policy_embedding_add = iam.ManagedPolicy(
            self,
            "PolicyLambdaEmbeddingAdd",
            managed_policy_name="EmbeddingsLake_LambdaEmbeddingAdd",
            document=iam.PolicyDocument(
                statements=[
                    iam.PolicyStatement(
                        effect=iam.Effect.ALLOW,
                        actions=[
                            "s3:PutObject",
                            "s3:HeadObject",
                            "s3:GetObject",
                        ],
                        resources=[
                            bucket_segments.bucket_arn,
                            bucket_segments.arn_for_objects("*"),
                        ],
                    )
                ]
            )
        )

        policy_embedding_table = iam.ManagedPolicy(
            self,
            "PolicyLambdaEmbeddingTable",
            managed_policy_name="EmbeddingsLake_LambdaEmbeddingTable",
            document=iam.PolicyDocument(
                statements=[
                    iam.PolicyStatement(
                        effect=iam.Effect.ALLOW,
                        actions=[
                            "dynamodb:PutItem",
                            "dynamodb:UpdateItem",
                            "dynamodb:BatchWriteItem"
                        ],
                        resources=[
                            table_embeddings.table_arn
                        ],
                    )
                ]
            )
        )

        policy_embedding_hash = iam.ManagedPolicy(
            self,
            "PolicyLambdaEmbeddingHash",
            managed_policy_name="EmbeddingsLake_LambdaEmbeddingHash",
            document=iam.PolicyDocument(
                statements=[
                    iam.PolicyStatement(
                        effect=iam.Effect.ALLOW,
                        actions=[
                            "s3:GetObject",
                            "s3:ListBucket"
                        ],
                        resources=[
                            bucket_segments.bucket_arn,
                            bucket_segments.arn_for_objects("*"),
                        ],
                    )
                ]
            )
        )

        policy_embedding_hash_add = iam.ManagedPolicy(
            self,
            "PolicyLambdaEmbeddingHashAdd",
            managed_policy_name="EmbeddingsLake_LambdaEmbeddingHashAdd",
            document=iam.PolicyDocument(
                statements=[
                    iam.PolicyStatement(
                        effect=iam.Effect.ALLOW,
                        actions=[
                            "s3:GetObject",
                            "s3:ListBucket"
                        ],
                        resources=[
                            bucket_segments.bucket_arn,
                            bucket_segments.arn_for_objects("*"),
                        ],
                    ),
                    iam.PolicyStatement(
                        effect=iam.Effect.ALLOW,
                        actions=[
                            "sqs:SendMessage"
                        ],
                        resources=[
                            queue_embeddings_add.queue_arn
                        ],
                    )
                ]
            )
        )

        policy_embedding_query = iam.ManagedPolicy(
            self,
            "PolicyLambdaEmbeddingQuery",
            managed_policy_name="EmbeddingsLake_LambdaEmbeddingQuery",
            document=iam.PolicyDocument(
                statements=[
                    iam.PolicyStatement(
                        effect=iam.Effect.ALLOW,
                        actions=[
                            "s3:GetObject",
                            "s3:ListBucket"
                        ],
                        resources=[
                            bucket_segments.bucket_arn,
                            bucket_segments.arn_for_objects("*"),
                        ],
                    )
                ]
            )
        )

        policy_embedding_adjacent = iam.ManagedPolicy(
            self,
            "PolicyLambdaEmbeddingAdjacent",
            managed_policy_name="EmbeddingsLake_LambdaEmbeddingAdjacent",
            document=iam.PolicyDocument(
                statements=[
                    iam.PolicyStatement(
                        effect=iam.Effect.ALLOW,
                        actions=[
                            "s3:GetObject",
                            "s3:ListBucket"
                        ],
                        resources=[
                            bucket_segments.bucket_arn,
                            bucket_segments.arn_for_objects("*"),
                        ],
                    )
                ]
            )
        )


        role_lambda_lake_instantiate = iam.Role(
            self,
            "RoleLambdaLakeInstantiation",
            assumed_by=iam.ServicePrincipal(lambda_endpoint),
            role_name="EmbeddingsLake_Role_lambda_Lake_Instantiation",
            managed_policies=[
                policy_lake_instantiate,
                iam.ManagedPolicy.from_aws_managed_policy_name("service-role/AWSLambdaBasicExecutionRole")
                ]
            
        )

        role_lambda_embedding_hash = iam.Role(
            self,
            "RoleLambdaEmbeddingHash",
            assumed_by=iam.ServicePrincipal(lambda_endpoint),
            role_name="EmbeddingsLake_Role_lambda_Embedding_Hash",
            managed_policies=[
                policy_embedding_hash,
                iam.ManagedPolicy.from_aws_managed_policy_name("service-role/AWSLambdaBasicExecutionRole")
                ]
            
        )

        role_lambda_embedding_hash_add = iam.Role(
            self,
            "RoleLambdaEmbeddingHashAdd",
            assumed_by=iam.ServicePrincipal(lambda_endpoint),
            role_name="EmbeddingsLake_Role_lambda_Embedding_HashAdd",
            managed_policies=[
                policy_embedding_hash_add,
                iam.ManagedPolicy.from_aws_managed_policy_name("service-role/AWSLambdaBasicExecutionRole")
                ]
            
        )

        role_lambda_embedding_add = iam.Role(
            self,
            "RoleLambdaEmbeddingAdd",
            assumed_by=iam.ServicePrincipal(lambda_endpoint),
            role_name="EmbeddingsLake_Role_lambda_Embedding_Add",
            managed_policies=[
                policy_embedding_add,
                iam.ManagedPolicy.from_aws_managed_policy_name("service-role/AWSLambdaBasicExecutionRole")
                ]
            
        )

        role_lambda_embedding_table = iam.Role(
            self,
            "RoleLambdaEmbeddingTable",
            assumed_by=iam.ServicePrincipal(lambda_endpoint),
            role_name="EmbeddingsLake_Role_lambda_Embedding_Table",
            managed_policies=[
                policy_embedding_table,
                iam.ManagedPolicy.from_aws_managed_policy_name("service-role/AWSLambdaBasicExecutionRole")
                ]
            
        )


        role_lambda_embedding_adjacent = iam.Role(
            self,
            "RoleLambdaEmbeddingAdjacent",
            assumed_by=iam.ServicePrincipal(lambda_endpoint),
            role_name="EmbeddingsLake_Role_lambda_Embedding_Adjacent",
            managed_policies=[
                policy_embedding_adjacent,
                iam.ManagedPolicy.from_aws_managed_policy_name("service-role/AWSLambdaBasicExecutionRole")
                ]
            
        )


        role_lambda_embedding_query = iam.Role(
            self,
            "RoleLambdaEmbeddingQuery",
            assumed_by=iam.ServicePrincipal(lambda_endpoint),
            role_name="EmbeddingsLake_Role_lambda_Embedding_Query",
            managed_policies=[
                policy_embedding_query,
                iam.ManagedPolicy.from_aws_managed_policy_name("service-role/AWSLambdaBasicExecutionRole")
                ]
            
        )

        role_lambda_embedding_sort = iam.Role(
            self,
            "RoleLambdaEmbeddingSort",
            assumed_by=iam.ServicePrincipal(lambda_endpoint),
            role_name="EmbeddingsLake_Role_lambda_Embedding_Sort",
            managed_policies=[
                iam.ManagedPolicy.from_aws_managed_policy_name("service-role/AWSLambdaBasicExecutionRole")
                ]
            
        )

        lambda_lake_instantiate = lambda_.Function(
            self,
            "FunctionInstantiateLake",
            runtime=lambda_.Runtime.PYTHON_3_10,
            handler="index.lambda_handler",
            code=lambda_.Code.from_asset("embeddings_lake/assets/lambda/laker"),
            environment={"BUCKET_NAME": bucket_segments.bucket_name },
            layers=[lambda_layer_pandas, lambda_layer_pydantic],
            role=role_lambda_lake_instantiate
        )

        lambda_embedding_hash_query = lambda_.Function(
            self,
            "FunctionEmbeddingHashQuery",
            runtime=lambda_.Runtime.PYTHON_3_10,
            handler="index.lambda_handler",
            code=lambda_.Code.from_asset("embeddings_lake/assets/lambda/hasher"),
            environment={"BUCKET_NAME": bucket_segments.bucket_name },
            layers=[lambda_layer_pandas],
            role=role_lambda_embedding_hash
        )

        lambda_embedding_hash_add = lambda_.Function(
            self,
            "FunctionEmbeddingHashAdd",
            runtime=lambda_.Runtime.PYTHON_3_10,
            memory_size=256,
            handler="index.lambda_handler",
            code=lambda_.Code.from_asset("embeddings_lake/assets/lambda/hashAdder"),
            environment={"BUCKET_NAME": bucket_segments.bucket_name, "QUEUE_URL": queue_embeddings_add.queue_url},
            layers=[lambda_layer_pandas],
            role=role_lambda_embedding_hash_add
        )

        lambda_embedding_add = lambda_.Function(
            self,
            "FunctionEmbeddingAdd",
            runtime=lambda_.Runtime.PYTHON_3_10,
            memory_size=1024,
            reserved_concurrent_executions=1,
            timeout=Duration.minutes(10),
            handler="index.lambda_handler",
            code=lambda_.Code.from_asset("embeddings_lake/assets/lambda/adder"),
            environment={"BUCKET_NAME": bucket_segments.bucket_name },
            layers=[lambda_layer_pandas, lambda_layer_pydantic],
            role=role_lambda_embedding_add,
            dead_letter_queue_enabled=True,
            dead_letter_queue=queue_dead_letter_embeddings_add
        )

        lambda_embedding_add.add_event_source(
            lambda_event_sources.SqsEventSource(
                queue=queue_embeddings_add,
                batch_size=1,
                #max_concurrency=2
            )
        )

        lambda_embedding_table = lambda_.Function(
            self,
            "FunctionEmbeddingTable",
            runtime=lambda_.Runtime.PYTHON_3_10,
            memory_size=1024,
            timeout=Duration.minutes(10),
            handler="index.lambda_handler",
            code=lambda_.Code.from_asset("embeddings_lake/assets/lambda/tabler"),
            environment={"TABLE_NAME": table_embeddings.table_name},
            #layers=[lambda_layer_pandas, lambda_layer_pydantic],
            role=role_lambda_embedding_table
        )

        lambda_embedding_adjacent = lambda_.Function(
            self,
            "FunctionEmbeddingAdjacent",
            runtime=lambda_.Runtime.PYTHON_3_10,
            timeout=Duration.minutes(1),
            handler="index.lambda_handler",
            code=lambda_.Code.from_asset("embeddings_lake/assets/lambda/adjacent"),
            environment={"BUCKET_NAME": bucket_segments.bucket_name },
            role=role_lambda_embedding_adjacent
        )

        lambda_embedding_query = lambda_.Function(
            self,
            "FunctionEmbeddingQuery",
            runtime=lambda_.Runtime.PYTHON_3_10,
            memory_size=1024,
            timeout=Duration.minutes(10),
            handler="index.lambda_handler",
            code=lambda_.Code.from_asset("embeddings_lake/assets/lambda/query"),
            environment={"BUCKET_NAME": bucket_segments.bucket_name },
            layers=[lambda_layer_pandas, lambda_layer_pydantic],
            role=role_lambda_embedding_query
        )

        lambda_embedding_sort = lambda_.Function(
            self,
            "FunctionEmbeddingSort",
            runtime=lambda_.Runtime.PYTHON_3_10,
            memory_size=256,
            timeout=Duration.minutes(1),
            handler="index.lambda_handler",
            code=lambda_.Code.from_asset("embeddings_lake/assets/lambda/sort"),
            environment={"BUCKET_NAME": bucket_segments.bucket_name },
            #layers=[lambda_layer_pandas, lambda_layer_pydantic],
            role=role_lambda_embedding_sort
        )

        task_embedding_hash = tasks.LambdaInvoke(
            self,
            "Hash Embedding",
            lambda_function=lambda_embedding_hash_query
        )

        task_embedding_add = tasks.LambdaInvoke(
            self,
            "Add Embedding",
            lambda_function=lambda_embedding_add,
        )

        task_embedding_table = tasks.LambdaInvoke(
            self,
            "Table Embedding",
            lambda_function=lambda_embedding_table,
        )



        task_embedding_adjacents = tasks.LambdaInvoke(
            self,
            "Get Adjacent Segments", 
            lambda_function=lambda_embedding_adjacent,

        )

        task_embedding_query = tasks.LambdaInvoke(
            self,
            "Search Segments",
            lambda_function=lambda_embedding_query
        )

        task_embedding_sort = tasks.LambdaInvoke(
            self,
            "Sort Segments",
            lambda_function=lambda_embedding_sort
        )

        choice_embedding = sfn.Choice(
            self,
            "Embedding Choice"
        )

        choice_embedding.when(
            condition=sfn.Condition.boolean_equals(variable="$.Payload.add", value=True),
            next=task_embedding_add
        )

        task_embedding_add.next(task_embedding_table)

        choice_embedding.otherwise(
            task_embedding_adjacents
        )      

        task_embedding_hash.next(choice_embedding)

        map_search_segments = sfn.Map(self, "Query Segments",
            max_concurrency=10,
            items_path="$.Payload.segmentIndices",
            input_path="$",
            parameters = { 
              "segmentIndex.$": "$$.Map.Item.Value",
              "embedding.$": "$.Payload.embedding",
              "lakeName.$": "$.Payload.lakeName",
              "distanceMetric.$": "$.Payload.distanceMetric"
            },
        )

        map_search_segments.item_processor(task_embedding_query)

        task_embedding_adjacents.next(map_search_segments)

        map_search_segments.next(task_embedding_sort)

        state_machine_embedding = sfn.StateMachine(
            self,
            id="StateMachineEmbeddingsLake",
            state_machine_type=sfn.StateMachineType.EXPRESS,
            logs=sfn.LogOptions(
                destination=logs.LogGroup(self, "MyLogGroup"),
                level=sfn.LogLevel.ALL
            ),
            definition=task_embedding_hash
        )

        api = apigateway.RestApi(
            self,
            "API Gateway",
            deploy_options=apigateway.StageOptions(
                stage_name="prod"
            )
        )

        api_integration_response_list = [
            apigateway.IntegrationResponse(status_code="200"),
            apigateway.IntegrationResponse(status_code="400"),
            apigateway.IntegrationResponse(status_code="500"),
        ]

        api_method_response_list = [
            apigateway.MethodResponse(status_code="200"),
            apigateway.MethodResponse(status_code="400"),
            apigateway.MethodResponse(status_code="500"),            
        ]


        api_resource_lake = api.root.add_resource("lake")

        api_resource_lake_embedding = api_resource_lake.add_resource("embedding")

        api_resource_lake_embedding_add = api_resource_lake_embedding.add_resource("add")

        api_resource_lake_embedding_query = api_resource_lake_embedding.add_resource("query")

        api_resource_lake.add_method(
            http_method="PUT",
            integration = apigateway.LambdaIntegration(
                    handler = lambda_lake_instantiate,
                    integration_responses=api_integration_response_list,
                    proxy=False
                    ),
            method_responses=api_method_response_list
        )

        api_resource_lake_embedding_add.add_method(
            http_method="PUT",
            integration = apigateway.LambdaIntegration(
                    handler = lambda_embedding_hash_add,
                    integration_responses=api_integration_response_list,
                    proxy=False
                    ),
            method_responses=api_method_response_list
            # integration = apigateway.StepFunctionsIntegration.start_execution(
            #     state_machine=state_machine_embedding
            #     )
        )

        api_resource_lake_embedding_query.add_method(
            http_method="PUT",
            # integration = apigateway.LambdaIntegration(
            #         handler = lambda_embedding_hash,
            #         integration_responses=api_integration_response_list,
            #         proxy=False
            #         ),
            # method_responses=api_method_response_list
            integration = apigateway.StepFunctionsIntegration.start_execution(
                state_machine=state_machine_embedding
                )
        )