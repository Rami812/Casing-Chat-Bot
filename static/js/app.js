// AI Case Interview Coach - Client-side JavaScript

let sessionId = null;
let isProcessing = false;

// API endpoints
const API = {
    startSession: '/start_session',
    upload: '/upload',
    chat: '/chat',
    resetSession: '/reset_session',
    health: '/health'
};

// Initialize session
async function startSession() {
    try {
        const response = await fetch(API.startSession, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            }
        });
        
        const data = await response.json();
        
        if (data.status === 'success') {
            sessionId = data.session_id;
            updateSessionStatus('Connected', 'success');
            console.log('Session started:', sessionId);
        } else {
            throw new Error(data.message || 'Failed to start session');
        }
    } catch (error) {
        console.error('Error starting session:', error);
        updateSessionStatus('Connection Failed', 'danger');
        showError('Failed to start session. Please refresh the page.');
    }
}

// Update session status indicator
function updateSessionStatus(status, type) {
    const statusElement = document.getElementById('session-status');
    if (statusElement) {
        const iconClass = type === 'success' ? 'bi-circle-fill text-success' : 
                         type === 'danger' ? 'bi-circle-fill text-danger' : 
                         'bi-circle-fill text-warning';
        
        statusElement.innerHTML = `<i class="bi ${iconClass} me-1"></i>${status}`;
    }
}

// Handle file upload
async function handleFileUpload(file) {
    if (!file) return;
    
    if (!file.type.includes('pdf')) {
        showError('Please upload a PDF file only.');
        return;
    }
    
    if (file.size > 20 * 1024 * 1024) { // 20MB
        showError('File size must be less than 20MB.');
        return;
    }
    
    const formData = new FormData();
    formData.append('file', file);
    
    try {
        showLoading('Processing PDF...', 'Extracting text and creating knowledge base...');
        
        const response = await fetch(API.upload, {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        
        if (data.status === 'success') {
            hideLoading();
            showSuccess(`File "${data.filename}" uploaded successfully!`);
            
            // Hide upload section and show chat
            document.getElementById('upload-section').style.display = 'none';
            document.getElementById('chat-section').style.display = 'block';
            
            // Add welcome message
            addMessage('assistant', 
                'Great! I\'ve processed your case study. I\'m ready to begin the interview. ' +
                'Type "start" or "begin" to receive the case prompt, or ask me any questions about the process.'
            );
            
            // Update stage
            updateCurrentStage('Stage 1 - Case Introduction', 'primary');
            
        } else {
            hideLoading();
            showError(data.error || 'Failed to upload file');
        }
    } catch (error) {
        hideLoading();
        console.error('Upload error:', error);
        showError('Error uploading file. Please try again.');
    }
}

// Send chat message
async function sendMessage() {
    const messageInput = document.getElementById('message-input');
    const message = messageInput.value.trim();
    
    if (!message || isProcessing) return;
    
    if (!sessionId) {
        showError('No active session. Please refresh the page.');
        return;
    }
    
    // Add user message to chat
    addMessage('user', message);
    messageInput.value = '';
    
    // Show typing indicator
    showTypingIndicator();
    isProcessing = true;
    
    try {
        const response = await fetch(API.chat, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ message: message })
        });
        
        const data = await response.json();
        
        hideTypingIndicator();
        isProcessing = false;
        
        if (data.response) {
            addMessage('assistant', data.response);
            
            // Update current stage if provided
            if (data.stage_display) {
                updateCurrentStage(data.stage_display, 'primary');
            }
        } else if (data.error) {
            showError(data.error);
        }
        
    } catch (error) {
        hideTypingIndicator();
        isProcessing = false;
        console.error('Chat error:', error);
        showError('Error sending message. Please try again.');
    }
}

// Add message to chat
function addMessage(sender, content) {
    const chatMessages = document.getElementById('chat-messages');
    const messageDiv = document.createElement('div');
    messageDiv.className = `chat-message ${sender} fade-in`;
    
    const isUser = sender === 'user';
    const avatarIcon = isUser ? 'bi-person-fill' : 'bi-robot';
    const slideClass = isUser ? 'slide-in-right' : 'slide-in-left';
    
    messageDiv.innerHTML = `
        <div class="message-avatar ${sender}">
            <i class="bi ${avatarIcon}"></i>
        </div>
        <div class="message-bubble ${sender} ${slideClass}">
            ${formatMessage(content)}
            <div class="message-timestamp">${getCurrentTime()}</div>
        </div>
    `;
    
    chatMessages.appendChild(messageDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

// Format message content
function formatMessage(content) {
    // Basic markdown-like formatting
    return content
        .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
        .replace(/\*(.*?)\*/g, '<em>$1</em>')
        .replace(/\n/g, '<br>')
        .replace(/- (.*?)(?=\n|$)/g, '• $1');
}

// Get current time
function getCurrentTime() {
    return new Date().toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'});
}

// Show typing indicator
function showTypingIndicator() {
    const chatMessages = document.getElementById('chat-messages');
    const typingDiv = document.createElement('div');
    typingDiv.id = 'typing-indicator';
    typingDiv.className = 'typing-indicator fade-in';
    typingDiv.innerHTML = `
        <div class="message-avatar assistant">
            <i class="bi bi-robot"></i>
        </div>
        <div class="typing-dots">
            <span></span>
            <span></span>
            <span></span>
        </div>
        <span class="ms-2">AI is thinking...</span>
    `;
    
    chatMessages.appendChild(typingDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

// Hide typing indicator
function hideTypingIndicator() {
    const typingIndicator = document.getElementById('typing-indicator');
    if (typingIndicator) {
        typingIndicator.remove();
    }
}

// Update current stage
function updateCurrentStage(stage, type = 'primary') {
    const stageElement = document.getElementById('current-stage');
    if (stageElement) {
        const badgeClass = `badge bg-${type} stage-indicator`;
        stageElement.className = badgeClass;
        stageElement.innerHTML = `<i class="bi bi-play-circle me-1"></i>${stage}`;
    }
}

// Reset session
async function resetSession() {
    if (!confirm('Are you sure you want to reset the session? This will clear all chat history and uploaded files.')) {
        return;
    }
    
    try {
        showLoading('Resetting session...', 'Please wait while we clean up and restart.');
        
        const response = await fetch(API.resetSession, {
            method: 'POST'
        });
        
        const data = await response.json();
        
        if (data.status === 'success') {
            hideLoading();
            location.reload(); // Reload the page to start fresh
        } else {
            hideLoading();
            showError('Failed to reset session');
        }
    } catch (error) {
        hideLoading();
        console.error('Reset error:', error);
        showError('Error resetting session. Please refresh the page manually.');
    }
}

// Show loading modal
function showLoading(title = 'Processing...', message = 'Please wait while we process your request.') {
    const modal = document.getElementById('loadingModal');
    const titleElement = modal.querySelector('h5');
    const messageElement = document.getElementById('loading-message');
    
    titleElement.textContent = title;
    messageElement.textContent = message;
    
    const bootstrapModal = new bootstrap.Modal(modal);
    bootstrapModal.show();
}

// Hide loading modal
function hideLoading() {
    const modal = document.getElementById('loadingModal');
    const bootstrapModal = bootstrap.Modal.getInstance(modal);
    if (bootstrapModal) {
        bootstrapModal.hide();
    }
}

// Show success message
function showSuccess(message) {
    showToast(message, 'success');
}

// Show error message
function showError(message) {
    showToast(message, 'danger');
}

// Show toast notification
function showToast(message, type = 'info') {
    // Create toast container if it doesn't exist
    let toastContainer = document.getElementById('toast-container');
    if (!toastContainer) {
        toastContainer = document.createElement('div');
        toastContainer.id = 'toast-container';
        toastContainer.className = 'toast-container position-fixed top-0 end-0 p-3';
        toastContainer.style.zIndex = '9999';
        document.body.appendChild(toastContainer);
    }
    
    // Create toast
    const toastId = 'toast-' + Date.now();
    const toastDiv = document.createElement('div');
    toastDiv.id = toastId;
    toastDiv.className = `toast align-items-center text-white bg-${type} border-0 fade-in`;
    toastDiv.setAttribute('role', 'alert');
    toastDiv.innerHTML = `
        <div class="d-flex">
            <div class="toast-body">
                ${message}
            </div>
            <button type="button" class="btn-close btn-close-white me-2 m-auto" 
                    data-bs-dismiss="toast"></button>
        </div>
    `;
    
    toastContainer.appendChild(toastDiv);
    
    // Show toast
    const toast = new bootstrap.Toast(toastDiv, {
        autohide: true,
        delay: type === 'danger' ? 8000 : 5000
    });
    toast.show();
    
    // Remove from DOM after hiding
    toastDiv.addEventListener('hidden.bs.toast', function() {
        toastDiv.remove();
    });
}

// Health check
async function checkHealth() {
    try {
        const response = await fetch(API.health);
        const data = await response.json();
        
        if (data.status === 'healthy') {
            console.log('Health check passed:', data);
            if (data.pinecone_available) {
                console.log('✅ Pinecone cloud storage available');
            } else {
                console.log('⚠️ Using local storage (Pinecone not available)');
            }
        }
    } catch (error) {
        console.error('Health check failed:', error);
    }
}

// Utility functions
function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}

// Event listeners
document.addEventListener('DOMContentLoaded', function() {
    // Perform health check
    checkHealth();
    
    // Setup periodic health checks
    setInterval(checkHealth, 5 * 60 * 1000); // Every 5 minutes
});

// Handle page unload
window.addEventListener('beforeunload', function(e) {
    if (isProcessing) {
        e.preventDefault();
        e.returnValue = 'A request is currently being processed. Are you sure you want to leave?';
        return e.returnValue;
    }
});

// Export functions for global access
window.startSession = startSession;
window.handleFileUpload = handleFileUpload;
window.sendMessage = sendMessage;
window.resetSession = resetSession;
window.showLoading = showLoading;
window.hideLoading = hideLoading;