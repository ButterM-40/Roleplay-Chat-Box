// REST API-based Chat Application
let currentCharacter = 'moses';
let conversationHistory = [];
let voiceEnabled = false; // Disabled by default
let voiceAvailable = false;
let currentAudio = null;

// Initialize the application
document.addEventListener('DOMContentLoaded', function() {
    setupCharacterSelection();
    setupEventListeners();
    loadSettings();
    checkVoiceStatus();
    showWelcomeMessage(currentCharacter);
    updateConnectionStatus(true); // Always connected in REST mode
});

function setupCharacterSelection() {
    const characterCards = document.querySelectorAll('.character-card');
    
    characterCards.forEach(card => {
        card.addEventListener('click', function() {
            const characterId = this.dataset.character;
            switchCharacter(characterId);
        });
    });
}

function setupEventListeners() {
    const messageInput = document.getElementById('message-input');
    const sendButton = document.getElementById('send-btn');
    
    messageInput.addEventListener('keydown', handleKeyPress);
    sendButton.addEventListener('click', sendMessage);
    
    // Auto-resize textarea
    messageInput.addEventListener('input', function() {
        autoResize(this);
    });
}

function handleKeyPress(event) {
    if (event.key === 'Enter' && !event.shiftKey) {
        event.preventDefault();
        sendMessage();
    }
}

function autoResize(textarea) {
    textarea.style.height = 'auto';
    textarea.style.height = Math.min(textarea.scrollHeight, 120) + 'px';
}

async function sendMessage() {
    const messageInput = document.getElementById('message-input');
    const message = messageInput.value.trim();
    
    if (!message) return;
    
    // Clear input and show user message
    messageInput.value = '';
    autoResize(messageInput);
    
    displayMessage(message, 'user');
    showTypingIndicator();
    
    try {
        // Send REST API request
        const requestData = {
            text: message,
            timestamp: Date.now(),
            conversation_history: conversationHistory.slice(-4), // Last 4 messages for context
            include_voice: voiceEnabled && voiceAvailable
        };
        
        console.log('Sending request:', {
            character: currentCharacter,
            includeVoice: requestData.include_voice,
            voiceEnabled: voiceEnabled,
            voiceAvailable: voiceAvailable
        });
        
        const response = await fetch(`/api/chat/${currentCharacter}`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(requestData)
        });
        
        hideTypingIndicator();
        
        if (response.ok) {
            const data = await response.json();
            console.log('Response data:', {
                hasVoiceData: !!data.voice_data,
                voiceEnabled: voiceEnabled,
                voiceAvailable: voiceAvailable,
                voiceDataLength: data.voice_data ? data.voice_data.length : 0
            });
            
            displayMessage(data.response, 'assistant', data.character_id);
            
            // Play voice if available
            if (data.voice_data && voiceEnabled) {
                console.log('Attempting to play voice...');
                playVoice(data.voice_data);
            } else if (!data.voice_data) {
                console.log('No voice data in response');
            } else if (!voiceEnabled) {
                console.log('Voice is disabled');
            }
            
            // Update conversation history
            conversationHistory.push(
                { role: 'user', content: message },
                { role: 'assistant', content: data.response }
            );
            
            // Limit history size
            if (conversationHistory.length > 20) {
                conversationHistory = conversationHistory.slice(-20);
            }
        } else {
            const error = await response.text();
            displayMessage(`Sorry, I encountered an error: ${error}`, 'error');
        }
        
    } catch (error) {
        hideTypingIndicator();
        console.error('Error sending message:', error);
        displayMessage('Sorry, I could not connect to the server. Please try again.', 'error');
    }
}

function displayMessage(content, type, characterId = null) {
    const chatMessages = document.getElementById('chat-messages');
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${type}`;
    
    if (type === 'user') {
        messageDiv.innerHTML = `
            <div class="message-avatar">
                <img src="/static/avatars/user.svg" alt="User" onerror="this.src='/static/avatars/default.svg'">
            </div>
            <div class="message-content">
                <div class="message-text">${escapeHtml(content)}</div>
                <div class="message-time">${new Date().toLocaleTimeString()}</div>
            </div>
        `;
    } else if (type === 'assistant') {
        const char = characterId || currentCharacter;
        const charName = getCharacterName(char);
        
        messageDiv.innerHTML = `
            <div class="message-avatar">
                <img src="/static/avatars/${char}.svg" alt="${charName}" onerror="this.src='/static/avatars/default.svg'">
            </div>
            <div class="message-content">
                <div class="message-text">${escapeHtml(content)}</div>
                <div class="message-time">${new Date().toLocaleTimeString()}</div>
            </div>
        `;
    } else if (type === 'error') {
        messageDiv.className = 'message error';
        messageDiv.innerHTML = `
            <div class="message-content">
                <div class="message-text error-text">❌ ${escapeHtml(content)}</div>
                <div class="message-time">${new Date().toLocaleTimeString()}</div>
            </div>
        `;
    }
    
    chatMessages.appendChild(messageDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

function getCharacterName(characterId) {
    const names = {
        'moses': 'Moses',
        'samsung_employee': 'Samsung Employee',
        'jinx': 'Jinx'
    };
    return names[characterId] || 'Character';
}

function switchCharacter(characterId) {
    // Update character selection UI
    document.querySelectorAll('.character-card').forEach(card => {
        card.classList.remove('active');
    });
    
    document.querySelector(`[data-character="${characterId}"]`).classList.add('active');
    
    currentCharacter = characterId;
    updateCharacterDisplay(characterId);
    
    // Clear current chat and show welcome message
    clearChat();
    showWelcomeMessage(characterId);
}

function updateCharacterDisplay(characterId) {
    const characterInfo = {
        'moses': {
            name: 'Moses',
            description: 'Biblical Prophet and Lawgiver',
            avatar: '/static/avatars/moses.svg'
        },
        'samsung_employee': {
            name: 'Samsung Employee',
            description: 'Tech-savvy Corporate Representative',
            avatar: '/static/avatars/samsung.svg'
        },
        'jinx': {
            name: 'Jinx',
            description: 'Chaotic Genius from Arcane',
            avatar: '/static/avatars/jinx.svg'
        }
    };
    
    const info = characterInfo[characterId];
    if (info) {
        document.getElementById('current-character-name').textContent = info.name;
        document.getElementById('current-character-desc').textContent = info.description;
        document.getElementById('current-avatar').src = info.avatar;
        
        // Update placeholder
        const messageInput = document.getElementById('message-input');
        messageInput.placeholder = `Message ${info.name}...`;
    }
}

function showWelcomeMessage(characterId) {
    const welcomeMessages = {
        'moses': "Peace be with you, my child. I am Moses, prophet and lawgiver. How may I guide you in righteousness?",
        'samsung_employee': "Hello! I'm your Samsung technology expert, ready to provide authentic product knowledge and enthusiasm. What amazing Galaxy features can I share with you today?",
        'jinx': "*spins around excitedly* Hey there! I'm Jinx - the real me, not some AI pretending! Pure chaotic genius with no boring assistant stuff. Ready for some explosive fun?"
    };
    
    const message = welcomeMessages[characterId];
    if (message) {
        displayMessage(message, 'assistant', characterId);
    }
}

function showTypingIndicator() {
    const indicator = document.getElementById('typing-indicator');
    const characterName = document.getElementById('typing-character');
    
    characterName.textContent = getCharacterName(currentCharacter);
    indicator.style.display = 'flex';
}

function hideTypingIndicator() {
    document.getElementById('typing-indicator').style.display = 'none';
}

function clearChat() {
    const chatMessages = document.getElementById('chat-messages');
    chatMessages.innerHTML = '';
    conversationHistory = [];
}

function startNewChat() {
    clearChat();
    showWelcomeMessage(currentCharacter);
}

function updateConnectionStatus(connected) {
    // In REST API mode, we're always "connected"
    const indicator = document.querySelector('.connection-status');
    if (indicator) {
        indicator.textContent = connected ? 'REST API Connected' : 'Disconnected';
        indicator.className = connected ? 'connection-status connected' : 'connection-status disconnected';
    }
}

function toggleVoice() {
    if (!voiceAvailable) {
        showNotification('Voice synthesis is not available on this server', 'error');
        return;
    }
    
    voiceEnabled = !voiceEnabled;
    updateVoiceIcon();
    
    // Save setting
    localStorage.setItem('voiceEnabled', voiceEnabled.toString());
    
    // Show notification
    const status = voiceEnabled ? 'enabled' : 'disabled';
    showNotification(`Voice output ${status}`, 'success');
}

function loadSettings() {
    // Load any saved settings
    const savedCharacter = localStorage.getItem('selectedCharacter');
    if (savedCharacter && ['moses', 'samsung_employee', 'jinx'].includes(savedCharacter)) {
        switchCharacter(savedCharacter);
    }
    
    // Load voice setting (default to false)
    const savedVoiceEnabled = localStorage.getItem('voiceEnabled');
    voiceEnabled = savedVoiceEnabled === 'true';
}

function showResources() {
    document.getElementById('resources-modal').style.display = 'flex';
}

function showSettings() {
    document.getElementById('settings-modal').style.display = 'flex';
}

function closeModal(modalId) {
    document.getElementById(modalId).style.display = 'none';
}

function changeTheme() {
    const theme = document.getElementById('theme-select').value;
    document.body.className = theme === 'light' ? 'light-theme' : '';
}

function updateVoiceSetting() {
    const checkbox = document.getElementById('voice-enabled');
    if (voiceAvailable) {
        checkbox.checked = voiceEnabled;
        checkbox.disabled = false;
        checkbox.onchange = function() {
            voiceEnabled = this.checked;
            updateVoiceIcon();
            localStorage.setItem('voiceEnabled', voiceEnabled.toString());
        };
    } else {
        checkbox.checked = false;
        checkbox.disabled = true;
    }
}

// Close modals when clicking outside
window.addEventListener('click', function(event) {
    const modals = document.querySelectorAll('.modal');
    modals.forEach(modal => {
        if (event.target === modal) {
            modal.style.display = 'none';
        }
    });
});

// Save character selection
window.addEventListener('beforeunload', function() {
    localStorage.setItem('selectedCharacter', currentCharacter);
});

// Voice-related functions
async function checkVoiceStatus() {
    try {
        const response = await fetch('/api/voice/status');
        if (response.ok) {
            const data = await response.json();
            voiceAvailable = data.voice_enabled && data.voice_model_loaded;
        } else {
            voiceAvailable = false;
        }
    } catch (error) {
        console.log('Voice status check failed:', error);
        voiceAvailable = false;
    }
    
    updateVoiceIcon();
    updateVoiceSetting();
}

function updateVoiceIcon() {
    const icon = document.getElementById('voice-icon');
    const button = icon.closest('.voice-toggle');
    
    if (!voiceAvailable) {
        icon.className = 'fas fa-volume-mute';
        button.classList.remove('active');
        button.title = 'Voice synthesis not available';
    } else if (voiceEnabled) {
        icon.className = 'fas fa-volume-up';
        button.classList.add('active');
        button.title = 'Voice enabled - Click to disable';
    } else {
        icon.className = 'fas fa-volume-off';
        button.classList.remove('active');
        button.title = 'Voice disabled - Click to enable';
    }
}

function playVoice(audioDataUrl) {
    try {
        console.log('Playing voice audio, data length:', audioDataUrl.length);
        console.log('Audio data preview:', audioDataUrl.substring(0, 50));
        
        // Stop any currently playing audio
        if (currentAudio) {
            currentAudio.pause();
            currentAudio.currentTime = 0;
        }
        
        // Create and play new audio
        currentAudio = new Audio(audioDataUrl);
        
        // Add event listeners for debugging
        currentAudio.addEventListener('loadstart', () => console.log('Audio loading started'));
        currentAudio.addEventListener('canplay', () => console.log('Audio can play'));
        currentAudio.addEventListener('playing', () => console.log('Audio is playing'));
        currentAudio.addEventListener('ended', () => console.log('Audio ended'));
        currentAudio.addEventListener('error', (e) => console.error('Audio error:', e));
        
        currentAudio.play().then(() => {
            console.log('Audio play() succeeded');
            showNotification('Voice playback started', 'success');
        }).catch(error => {
            console.error('Error playing voice:', error);
            showNotification('Failed to play voice audio: ' + error.message, 'error');
        });
    } catch (error) {
        console.error('Error setting up voice playback:', error);
        showNotification('Voice playback error: ' + error.message, 'error');
    }
}

function showNotification(message, type = 'info') {
    // Create notification element
    const notification = document.createElement('div');
    notification.className = `notification ${type}`;
    notification.innerHTML = `
        <i class="fas ${
            type === 'success' ? 'fa-check-circle' : 
            type === 'error' ? 'fa-exclamation-circle' : 'fa-info-circle'
        }"></i>
        <span>${message}</span>
        <button onclick="this.parentElement.remove()">
            <i class="fas fa-times"></i>
        </button>
    `;
    
    // Add to page
    document.body.appendChild(notification);
    
    // Auto remove after 3 seconds
    setTimeout(() => {
        if (notification.parentElement) {
            notification.remove();
        }
    }, 3000);
}

// Add connection status indicator
document.addEventListener('DOMContentLoaded', function() {
    const header = document.querySelector('.chat-header');
    if (header && !document.querySelector('.connection-status')) {
        const statusDiv = document.createElement('div');
        statusDiv.className = 'connection-status connected';
        statusDiv.textContent = 'REST API Ready';
        statusDiv.style.cssText = 'font-size: 12px; color: #4CAF50; margin-left: 10px;';
        header.appendChild(statusDiv);
    }
});