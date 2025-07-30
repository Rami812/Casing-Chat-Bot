"""
AWS Services Manager for Case Interview Application
Handles S3, DynamoDB, and other AWS service integrations for free tier usage
"""

import os
import json
import asyncio
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Union
import boto3
from botocore.exceptions import ClientError, NoCredentialsError
import aiobotocore.session
from fastapi import HTTPException
import logging

from .config import get_settings

logger = logging.getLogger(__name__)

class AWSManager:
    """Manages AWS services integration for the application"""
    
    def __init__(self):
        self.settings = get_settings()
        self.session = None
        self.s3_client = None
        self.dynamodb_client = None
        self.dynamodb_resource = None
        self._initialized = False
        
    async def initialize(self):
        """Initialize AWS clients and verify connections"""
        try:
            # Create aiobotocore session for async operations
            self.session = aiobotocore.session.get_session()
            
            # Initialize clients
            await self._init_s3()
            await self._init_dynamodb()
            
            # Verify AWS connectivity
            await self._verify_aws_connection()
            
            # Create necessary resources if they don't exist
            await self._setup_aws_resources()
            
            self._initialized = True
            logger.info("✅ AWS Manager initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize AWS Manager: {str(e)}")
            raise HTTPException(status_code=500, detail=f"AWS initialization failed: {str(e)}")
    
    async def _init_s3(self):
        """Initialize S3 client"""
        self.s3_client = self.session.create_client(
            's3',
            region_name=self.settings.AWS_REGION,
            aws_access_key_id=self.settings.AWS_ACCESS_KEY_ID,
            aws_secret_access_key=self.settings.AWS_SECRET_ACCESS_KEY
        )
    
    async def _init_dynamodb(self):
        """Initialize DynamoDB client and resource"""
        self.dynamodb_client = self.session.create_client(
            'dynamodb',
            region_name=self.settings.AWS_REGION,
            aws_access_key_id=self.settings.AWS_ACCESS_KEY_ID,
            aws_secret_access_key=self.settings.AWS_SECRET_ACCESS_KEY
        )
        
        # For high-level operations
        self.dynamodb_resource = self.session.create_resource(
            'dynamodb',
            region_name=self.settings.AWS_REGION,
            aws_access_key_id=self.settings.AWS_ACCESS_KEY_ID,
            aws_secret_access_key=self.settings.AWS_SECRET_ACCESS_KEY
        )
    
    async def _verify_aws_connection(self):
        """Verify AWS connection and credentials"""
        try:
            # Test S3 connection
            async with self.s3_client as s3:
                await s3.list_buckets()
            
            # Test DynamoDB connection
            async with self.dynamodb_client as dynamodb:
                await dynamodb.list_tables()
                
            logger.info("✅ AWS connectivity verified")
            
        except NoCredentialsError:
            raise HTTPException(status_code=500, detail="AWS credentials not found")
        except ClientError as e:
            raise HTTPException(status_code=500, detail=f"AWS connection failed: {str(e)}")
    
    async def _setup_aws_resources(self):
        """Create necessary AWS resources if they don't exist"""
        # Create S3 bucket
        await self._create_s3_bucket()
        
        # Create DynamoDB tables
        await self._create_dynamodb_tables()
    
    async def _create_s3_bucket(self):
        """Create S3 bucket if it doesn't exist"""
        try:
            async with self.s3_client as s3:
                try:
                    await s3.head_bucket(Bucket=self.settings.S3_BUCKET_NAME)
                    logger.info(f"✅ S3 bucket {self.settings.S3_BUCKET_NAME} already exists")
                except ClientError as e:
                    if e.response['Error']['Code'] == '404':
                        # Bucket doesn't exist, create it
                        await s3.create_bucket(Bucket=self.settings.S3_BUCKET_NAME)
                        logger.info(f"✅ Created S3 bucket: {self.settings.S3_BUCKET_NAME}")
                    else:
                        raise
        except Exception as e:
            logger.error(f"❌ Failed to create S3 bucket: {str(e)}")
            raise
    
    async def _create_dynamodb_tables(self):
        """Create DynamoDB tables if they don't exist"""
        tables_config = [
            {
                'TableName': self.settings.DYNAMODB_TABLE_SESSIONS,
                'KeySchema': [{'AttributeName': 'session_id', 'KeyType': 'HASH'}],
                'AttributeDefinitions': [{'AttributeName': 'session_id', 'AttributeType': 'S'}],
                'BillingMode': 'PAY_PER_REQUEST'
            },
            {
                'TableName': self.settings.DYNAMODB_TABLE_USERS,
                'KeySchema': [{'AttributeName': 'user_id', 'KeyType': 'HASH'}],
                'AttributeDefinitions': [{'AttributeName': 'user_id', 'AttributeType': 'S'}],
                'BillingMode': 'PAY_PER_REQUEST'
            },
            {
                'TableName': self.settings.DYNAMODB_TABLE_ANALYTICS,
                'KeySchema': [
                    {'AttributeName': 'session_id', 'KeyType': 'HASH'},
                    {'AttributeName': 'timestamp', 'KeyType': 'RANGE'}
                ],
                'AttributeDefinitions': [
                    {'AttributeName': 'session_id', 'AttributeType': 'S'},
                    {'AttributeName': 'timestamp', 'AttributeType': 'S'}
                ],
                'BillingMode': 'PAY_PER_REQUEST'
            }
        ]
        
        async with self.dynamodb_client as dynamodb:
            for table_config in tables_config:
                try:
                    await dynamodb.describe_table(TableName=table_config['TableName'])
                    logger.info(f"✅ DynamoDB table {table_config['TableName']} already exists")
                except ClientError as e:
                    if e.response['Error']['Code'] == 'ResourceNotFoundException':
                        await dynamodb.create_table(**table_config)
                        logger.info(f"✅ Created DynamoDB table: {table_config['TableName']}")
                    else:
                        raise
    
    # S3 Operations
    async def upload_file_to_s3(self, file_content: bytes, file_key: str, content_type: str = "application/pdf") -> str:
        """Upload file to S3 and return the file URL"""
        try:
            async with self.s3_client as s3:
                await s3.put_object(
                    Bucket=self.settings.S3_BUCKET_NAME,
                    Key=file_key,
                    Body=file_content,
                    ContentType=content_type
                )
            
            file_url = f"https://{self.settings.S3_BUCKET_NAME}.s3.{self.settings.AWS_REGION}.amazonaws.com/{file_key}"
            logger.info(f"✅ File uploaded to S3: {file_key}")
            return file_url
            
        except Exception as e:
            logger.error(f"❌ Failed to upload file to S3: {str(e)}")
            raise HTTPException(status_code=500, detail=f"File upload failed: {str(e)}")
    
    async def download_file_from_s3(self, file_key: str) -> bytes:
        """Download file from S3"""
        try:
            async with self.s3_client as s3:
                response = await s3.get_object(Bucket=self.settings.S3_BUCKET_NAME, Key=file_key)
                return await response['Body'].read()
        except Exception as e:
            logger.error(f"❌ Failed to download file from S3: {str(e)}")
            raise HTTPException(status_code=404, detail=f"File not found: {file_key}")
    
    async def delete_file_from_s3(self, file_key: str) -> bool:
        """Delete file from S3"""
        try:
            async with self.s3_client as s3:
                await s3.delete_object(Bucket=self.settings.S3_BUCKET_NAME, Key=file_key)
            logger.info(f"✅ File deleted from S3: {file_key}")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to delete file from S3: {str(e)}")
            return False
    
    # DynamoDB Operations
    async def save_session_data(self, session_id: str, session_data: Dict[str, Any]) -> bool:
        """Save session data to DynamoDB"""
        try:
            item = {
                'session_id': {'S': session_id},
                'data': {'S': json.dumps(session_data)},
                'timestamp': {'S': datetime.now(timezone.utc).isoformat()},
                'ttl': {'N': str(int(datetime.now().timestamp()) + 86400)}  # 24h TTL
            }
            
            async with self.dynamodb_client as dynamodb:
                await dynamodb.put_item(
                    TableName=self.settings.DYNAMODB_TABLE_SESSIONS,
                    Item=item
                )
            
            logger.info(f"✅ Session data saved: {session_id}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to save session data: {str(e)}")
            return False
    
    async def get_session_data(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get session data from DynamoDB"""
        try:
            async with self.dynamodb_client as dynamodb:
                response = await dynamodb.get_item(
                    TableName=self.settings.DYNAMODB_TABLE_SESSIONS,
                    Key={'session_id': {'S': session_id}}
                )
            
            if 'Item' in response:
                return json.loads(response['Item']['data']['S'])
            return None
            
        except Exception as e:
            logger.error(f"❌ Failed to get session data: {str(e)}")
            return None
    
    async def save_analytics_data(self, session_id: str, event_type: str, data: Dict[str, Any]) -> bool:
        """Save analytics data to DynamoDB"""
        try:
            timestamp = datetime.now(timezone.utc).isoformat()
            item = {
                'session_id': {'S': session_id},
                'timestamp': {'S': timestamp},
                'event_type': {'S': event_type},
                'data': {'S': json.dumps(data)},
                'ttl': {'N': str(int(datetime.now().timestamp()) + 2592000)}  # 30 days TTL
            }
            
            async with self.dynamodb_client as dynamodb:
                await dynamodb.put_item(
                    TableName=self.settings.DYNAMODB_TABLE_ANALYTICS,
                    Item=item
                )
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to save analytics data: {str(e)}")
            return False
    
    async def health_check(self) -> Dict[str, Any]:
        """Check the health of AWS services"""
        health_status = {
            "s3": False,
            "dynamodb": False,
            "overall": False
        }
        
        try:
            # Check S3
            async with self.s3_client as s3:
                await s3.head_bucket(Bucket=self.settings.S3_BUCKET_NAME)
            health_status["s3"] = True
            
            # Check DynamoDB
            async with self.dynamodb_client as dynamodb:
                await dynamodb.describe_table(TableName=self.settings.DYNAMODB_TABLE_SESSIONS)
            health_status["dynamodb"] = True
            
            health_status["overall"] = health_status["s3"] and health_status["dynamodb"]
            
        except Exception as e:
            logger.error(f"❌ AWS health check failed: {str(e)}")
        
        return health_status