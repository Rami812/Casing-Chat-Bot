# AI Case Interview Coach - Flask Web Application

A comprehensive AI-powered case interview practice platform with cloud-based knowledge storage, built with Flask and multiple AI agents.

## 🚀 Features

- **Multi-Agent AI System**: Powered by CrewAI with specialized agents for different interview stages
- **Cloud Knowledge Base**: Uses Pinecone vector database for scalable document storage
- **Real-time Chat Interface**: Modern web interface with responsive design
- **PDF Document Processing**: Upload and process case study PDFs automatically
- **Stage-based Interview Flow**: Guided progression through case interview stages
- **Session Management**: Secure session handling with cleanup capabilities

## 🏗️ Architecture

- **Backend**: Flask web application with RESTful API
- **Frontend**: Bootstrap 5 with vanilla JavaScript
- **AI Agents**: CrewAI framework with Google Gemini LLM
- **Vector Database**: Pinecone for cloud-based knowledge storage
- **Fallback Storage**: Local FAISS for development/testing

## 📋 Prerequisites

- Python 3.11+
- Google API Key (for Gemini)
- Pinecone API Key (for cloud storage)
- Docker (optional, for containerized deployment)

## 🛠️ Installation

### Local Development

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd ai-case-interview-coach
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   ```bash
   cp .env.example .env
   # Edit .env file with your API keys
   ```

5. **Run the application**
   ```bash
   python flask_app.py
   ```

### Docker Deployment

1. **Build and run with Docker Compose**
   ```bash
   cp .env.example .env
   # Edit .env file with your API keys
   docker-compose up --build
   ```

2. **Access the application**
   - Open your browser to `http://localhost:5000`

## 🔧 Configuration

### Environment Variables

Create a `.env` file with the following variables:

```env
# Core Application Settings
FLASK_SECRET_KEY=your-super-secret-key-change-this-in-production
FLASK_DEBUG=False
HOST=0.0.0.0
PORT=5000

# AI Configuration
GOOGLE_API_KEY=your-google-gemini-api-key-here

# Cloud Storage
PINECONE_API_KEY=your-pinecone-api-key-here
```

### Getting API Keys

1. **Google Gemini API Key**:
   - Go to [Google AI Studio](https://makersuite.google.com/app/apikey)
   - Create a new API key
   - Add it to your `.env` file

2. **Pinecone API Key**:
   - Sign up at [Pinecone](https://www.pinecone.io/)
   - Create a new project
   - Get your API key from the dashboard
   - Add it to your `.env` file

## 🎯 Usage

### Interview Process

1. **Upload Case Study**: Upload a PDF case study document
2. **Case Introduction**: AI presents the case scenario
3. **Clarifying Questions**: Ask strategic questions about the business context
4. **Framework Development**: Create and refine your analytical framework
5. **Quantitative Analysis**: Perform calculations and mathematical analysis
6. **Recommendations**: Present final recommendations and next steps

### Stage Progression

The AI automatically detects and transitions between stages based on your responses:
- Use keywords like "framework" or "approach" to move to framework evaluation
- Include calculations or numbers to trigger math evaluation stage
- Use words like "recommend" or "conclusion" for final recommendations

## 🚀 Deployment

### Cloud Platforms

#### Heroku
```bash
# Install Heroku CLI
heroku create your-app-name
heroku config:set GOOGLE_API_KEY=your-key
heroku config:set PINECONE_API_KEY=your-key
heroku config:set FLASK_SECRET_KEY=your-secret-key
git push heroku main
```

#### Google Cloud Platform
```bash
# Using Cloud Run
gcloud run deploy --source . --platform managed --region us-central1
```

#### AWS (using Docker)
```bash
# Build and push to ECR
aws ecr create-repository --repository-name ai-case-coach
docker build -t ai-case-coach .
docker tag ai-case-coach:latest <account-id>.dkr.ecr.us-east-1.amazonaws.com/ai-case-coach:latest
docker push <account-id>.dkr.ecr.us-east-1.amazonaws.com/ai-case-coach:latest
```

### Production Considerations

1. **Security**:
   - Use strong, random secret keys
   - Enable HTTPS in production
   - Consider adding rate limiting

2. **Scaling**:
   - Use multiple Gunicorn workers
   - Consider Redis for session storage
   - Monitor Pinecone usage and costs

3. **Monitoring**:
   - Implement logging
   - Add error tracking (e.g., Sentry)
   - Monitor API usage and costs

## 🔌 API Endpoints

- `GET /` - Main application page
- `POST /start_session` - Initialize new interview session
- `POST /upload` - Upload PDF case study
- `POST /chat` - Send chat message and get AI response
- `POST /reset_session` - Reset current session
- `GET /health` - Health check endpoint

## 🧪 Development

### Project Structure
```
ai-case-interview-coach/
├── flask_app.py           # Main Flask application
├── templates/             # HTML templates
│   ├── base.html         # Base template
│   └── index.html        # Main page template
├── static/               # Static files
│   ├── css/style.css     # Custom styles
│   └── js/app.js         # Client-side JavaScript
├── requirements.txt      # Python dependencies
├── Dockerfile           # Docker configuration
├── docker-compose.yml   # Docker Compose setup
└── README.md           # This file
```

### Adding New Features

1. **New AI Agents**: Extend the `CaseInterviewCrew` class
2. **Additional Stages**: Add new task types and detection logic
3. **UI Enhancements**: Modify templates and static files
4. **New Endpoints**: Add routes to `flask_app.py`

## 🐛 Troubleshooting

### Common Issues

1. **Pinecone Connection Failed**:
   - Check your API key
   - Verify your Pinecone project is active
   - Application will fallback to local FAISS storage

2. **PDF Processing Errors**:
   - Ensure PDF is not password-protected
   - Check file size (max 20MB)
   - Verify PDF contains extractable text

3. **AI Response Errors**:
   - Verify Google API key is valid
   - Check API usage limits
   - Monitor error logs for specific issues

### Logs and Debugging

- Enable Flask debug mode: `FLASK_DEBUG=True`
- Check browser console for JavaScript errors
- Monitor server logs for backend issues
- Use health endpoint for system status

## 📞 Support

For issues and questions:
1. Check the troubleshooting section
2. Review error logs and console output
3. Verify all API keys are correctly configured
4. Ensure all dependencies are properly installed

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

---

**Built with ❤️ using Flask, CrewAI, and cloud technologies**