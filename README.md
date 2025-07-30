# AI Case Interview Coach - FastAPI Edition

A modern, cloud-powered case interview practice application built with FastAPI and AWS services. This application provides an AI-powered coaching system that guides users through all stages of case interview preparation with multi-agent CrewAI intelligence.

## 🚀 Features

### Core Functionality
- **Multi-Agent AI Coaching**: CrewAI-powered system with specialized agents for each interview stage
- **PDF Case Study Processing**: Upload and analyze case studies with vector search capabilities
- **Real-time Communication**: WebSocket-based chat with instant AI feedback
- **Stage-based Interview Flow**: Guided progression through 5 interview stages
- **Session Persistence**: AWS DynamoDB-backed session storage with analytics

### Technical Improvements
- **FastAPI Backend**: Modern, async Python web framework with automatic API documentation
- **AWS Cloud Integration**: S3 for file storage, DynamoDB for data persistence
- **Real-time WebSockets**: Instant communication with typing indicators and connection status
- **Modern Frontend**: Responsive UI with Tailwind CSS and real-time updates
- **Scalable Architecture**: Cloud-ready with proper error handling and logging

## 🏗️ Architecture

```
Frontend (HTML/JS/Tailwind) 
    ↓ WebSocket & REST API
FastAPI Backend
    ↓ 
CrewAI Multi-Agent System
    ↓
AWS Services (S3, DynamoDB)
```

### Interview Stages
1. **Case Introduction** - Present case scenario and validate understanding
2. **Clarifying Questions** - Evaluate strategic question quality
3. **Framework Development** - Check MECE frameworks and structure
4. **Quantitative Analysis** - Verify calculations and business math
5. **Recommendations** - Assess final presentation and next steps

## 🛠️ AWS Setup (Free Tier)

### 1. Create AWS Account
- Sign up for AWS Free Tier account
- Navigate to IAM and create a new user with programmatic access

### 2. Configure S3 (Storage)
```bash
# Create S3 bucket for file storage
aws s3 mb s3://your-case-interview-files
```

### 3. Configure DynamoDB (Database)
```bash
# Tables will be created automatically on first run
# Or create manually:
aws dynamodb create-table \
    --table-name case-interview-sessions \
    --attribute-definitions AttributeName=session_id,AttributeType=S \
    --key-schema AttributeName=session_id,KeyType=HASH \
    --billing-mode PAY_PER_REQUEST
```

### 4. IAM Permissions
Create an IAM policy with these permissions:
```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": [
                "s3:GetObject",
                "s3:PutObject",
                "s3:DeleteObject",
                "s3:ListBucket"
            ],
            "Resource": [
                "arn:aws:s3:::your-bucket-name",
                "arn:aws:s3:::your-bucket-name/*"
            ]
        },
        {
            "Effect": "Allow",
            "Action": [
                "dynamodb:GetItem",
                "dynamodb:PutItem",
                "dynamodb:UpdateItem",
                "dynamodb:DeleteItem",
                "dynamodb:Query",
                "dynamodb:Scan",
                "dynamodb:CreateTable",
                "dynamodb:DescribeTable"
            ],
            "Resource": "arn:aws:dynamodb:*:*:table/case-interview-*"
        }
    ]
}
```

## 🚀 Installation & Setup

### 1. Clone Repository
```bash
git clone <repository-url>
cd case-interview-fastapi
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Environment Configuration
```bash
# Copy environment template
cp .env.example .env

# Edit .env with your credentials
# Required variables:
# - GOOGLE_API_KEY (for Gemini AI)
# - AWS_ACCESS_KEY_ID
# - AWS_SECRET_ACCESS_KEY
# - S3_BUCKET_NAME
```

### 4. Run Application
```bash
# Development mode
python main.py

# Or with uvicorn
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### 5. Access Application
- **Frontend**: http://localhost:8000
- **API Documentation**: http://localhost:8000/api/docs
- **Health Check**: http://localhost:8000/health

## 📡 API Endpoints

### Session Management
- `POST /api/auth/guest-session` - Create guest session
- `POST /api/case/session/create` - Create new case session
- `GET /api/case/session/{session_id}` - Get session info

### Case Interview
- `POST /api/case/session/{session_id}/upload` - Upload PDF case study
- `POST /api/case/session/{session_id}/message` - Send message to AI
- `GET /api/case/session/{session_id}/history` - Get chat history
- `GET /api/case/session/{session_id}/analytics` - Get session analytics

### WebSocket
- `WS /ws/chat/{session_id}` - Real-time chat connection

## 🔧 Configuration Options

### File Upload Settings
```python
MAX_FILE_SIZE = 20MB  # Maximum PDF size
ALLOWED_FILE_TYPES = ["application/pdf"]
CHUNK_SIZE = 1000  # Text chunking for vector search
CHUNK_OVERLAP = 200  # Overlap between chunks
```

### AWS Settings
```python
AWS_REGION = "us-east-1"  # AWS region
S3_BUCKET_NAME = "case-interview-files"  # S3 bucket name
DYNAMODB_TABLE_SESSIONS = "case-interview-sessions"  # Session table
```

## 🎯 Usage Guide

### For Students
1. **Start Session**: Open application and create new session
2. **Upload Case**: Upload PDF case study (max 20MB)
3. **Begin Interview**: Type "start" to receive case prompt
4. **Progress Through Stages**:
   - Ask clarifying questions
   - Present your framework
   - Perform calculations
   - Make recommendations
5. **Get Feedback**: Receive expert feedback at each stage

### For Developers
1. **API Integration**: Use REST endpoints for custom frontends
2. **WebSocket Events**: Handle real-time messaging
3. **AWS Monitoring**: Monitor usage in AWS CloudWatch
4. **Analytics**: Track user behavior via DynamoDB analytics

## 🔍 Monitoring & Analytics

### Application Health
```bash
curl http://localhost:8000/health
```

### AWS CloudWatch Metrics
- S3 upload/download statistics
- DynamoDB read/write capacity
- API response times
- Error rates

### Session Analytics
```bash
curl http://localhost:8000/api/case/session/{session_id}/analytics
```

## 🚀 Production Deployment

### Docker Deployment
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### AWS ECS/Fargate
1. Build and push Docker image to ECR
2. Create ECS task definition
3. Deploy to Fargate with load balancer
4. Configure environment variables

### Environment Variables for Production
```bash
DEBUG=false
HOST=0.0.0.0
PORT=8000
SECRET_KEY=production-secret-key
# ... other production configs
```

## 🛡️ Security Considerations

1. **API Keys**: Store in AWS Secrets Manager for production
2. **CORS**: Configure specific origins instead of wildcard
3. **Rate Limiting**: Implement request rate limiting
4. **Input Validation**: All file uploads are validated
5. **Session Security**: Session IDs are UUIDs with proper cleanup

## 📊 Cost Optimization (AWS Free Tier)

### S3 Storage
- 5GB free storage
- 20,000 GET requests
- 2,000 PUT requests

### DynamoDB
- 25GB free storage
- 25 read/write capacity units

### Monitoring
- Basic CloudWatch metrics included
- Set up billing alerts for cost control

## 🐛 Troubleshooting

### Common Issues

1. **AWS Credentials Error**
   ```bash
   # Verify AWS credentials
   aws configure list
   ```

2. **S3 Bucket Permissions**
   ```bash
   # Check bucket policy
   aws s3api get-bucket-policy --bucket your-bucket-name
   ```

3. **DynamoDB Table Not Found**
   ```bash
   # List tables
   aws dynamodb list-tables
   ```

4. **WebSocket Connection Issues**
   - Check firewall settings
   - Verify WebSocket URL format
   - Check browser console for errors

### Debug Mode
```bash
# Enable debug logging
DEBUG=true uvicorn main:app --reload --log-level debug
```

## 🤝 Contributing

1. Fork the repository
2. Create feature branch: `git checkout -b feature/amazing-feature`
3. Commit changes: `git commit -m 'Add amazing feature'`
4. Push to branch: `git push origin feature/amazing-feature`
5. Open Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **CrewAI**: Multi-agent AI framework
- **FastAPI**: Modern Python web framework
- **AWS**: Cloud infrastructure services
- **Tailwind CSS**: Utility-first CSS framework

---

## 🎓 Comparison: Chainlit vs FastAPI

| Feature | Original (Chainlit) | Improved (FastAPI) |
|---------|-------------------|-------------------|
| Framework | Chainlit | FastAPI + Custom Frontend |
| Real-time | Built-in chat | WebSocket + REST API |
| Storage | In-memory | AWS S3 + DynamoDB |
| Scalability | Single instance | Cloud-native, multi-instance |
| API | Limited | Full REST API + WebSocket |
| UI/UX | Basic chat | Modern responsive design |
| Analytics | None | Built-in with AWS |
| Deployment | Simple | Production-ready |
| Cost | Local only | AWS Free Tier compatible |

The FastAPI version provides enterprise-grade scalability, cloud persistence, and a modern user experience while maintaining all the original AI coaching functionality.