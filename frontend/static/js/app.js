// WebSocket connection
let ws = null;
let clientId = generateClientId();
let currentCharacter = 'moses';
let conversationHistory = [];
let voiceEnabled = true;

// Initialize the application
document.addEventListener('DOMContentLoaded', function() {
    initializeWebSocket();
    setupCharacterSelection();
    setupEventListeners();
    loadSettings();
});

function generateClientId() {
    return 'client_' + Math.random().toString(36).substr(2, 9);
}

function initializeWebSocket() {
    const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${wsProtocol}//${window.location.host}/ws/${clientId}`;
    
    ws = new WebSocket(wsUrl);
    
    ws.onopen = function() {
        console.log('WebSocket connected');
        updateConnectionStatus(true);
    };
    
    ws.onmessage = function(event) {
        const data = JSON.parse(event.data);
        handleWebSocketMessage(data);
    };
    
    ws.onclose = function() {
        console.log('WebSocket disconnected');
        updateConnectionStatus(false);
        
        // Attempt to reconnect after 3 seconds
        setTimeout(initializeWebSocket, 3000);
    };
    
    ws.onerror = function(error) {
        console.error('WebSocket error:', error);
        updateConnectionStatus(false);
    };
}

function updateConnectionStatus(connected) {
    // You could add a connection indicator in the UI
    const sendBtn = document.getElementById('send-btn');
    sendBtn.disabled = !connected;
}

function handleWebSocketMessage(data) {
    switch(data.type) {
        case 'character_switched':
            console.log(`Switched to character: ${data.character_id}`);
            break;
            
        case 'chat_response':
            hideTypingIndicator();
            displayMessage(data.response, 'assistant', data.character_id);
            
            // Play audio if available
            if (data.audio && voiceEnabled) {
                playAudio(data.audio);
            }
            break;
    }
}

function setupCharacterSelection() {
    const characterCards = document.querySelectorAll('.character-card');
    
    characterCards.forEach(card => {
        card.addEventListener('click', function() {
            const characterId = this.dataset.character;
            switchCharacter(characterId);
            
            // Show enhancement notification
            showEnhancementNotification(characterId);
        });
    });
    
    // Set initial character as active
    setActiveCharacter(currentCharacter);
}

function showEnhancementNotification(character) {
    const notifications = {
        'moses': 'Moses enhanced with 70 examples of divine wisdom and biblical knowledge',
        'samsung_employee': 'Samsung Employee enhanced with 60 examples of technical expertise',
        'jinx': 'Jinx enhanced with 60 examples of chaotic personality and emotional depth'
    };
    
    const message = notifications[character] || 'Character enhanced with 5x training data';
    showNotification(message, 'enhancement');
}

function showNotification(message, type = 'info') {
    // Remove existing notifications
    document.querySelectorAll('.notification').forEach(n => n.remove());
    
    const notification = document.createElement('div');
    notification.className = `notification ${type}`;
    notification.innerHTML = `
        <i class="fas fa-star"></i>
        <span>${message}</span>
        <button onclick="this.parentElement.remove()">×</button>
    `;
    
    document.body.appendChild(notification);
    
    // Auto remove after 5 seconds
    setTimeout(() => {
        notification.remove();
    }, 5000);
}

function setupEventListeners() {
    const messageInput = document.getElementById('message-input');
    const sendBtn = document.getElementById('send-btn');
    
    // Send message on button click
    sendBtn.addEventListener('click', sendMessage);
    
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

function switchCharacter(characterId) {
    if (currentCharacter === characterId) return;
    
    currentCharacter = characterId;
    setActiveCharacter(characterId);
    updateChatHeader(characterId);
    
    // Send switch message via WebSocket
    if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({
            type: 'switch_character',
            character_id: characterId
        }));
    }
    
    // Clear current chat and show welcome message
    clearChat();
    showWelcomeMessage(characterId);
}

function setActiveCharacter(characterId) {
    const characterCards = document.querySelectorAll('.character-card');
    
    characterCards.forEach(card => {
        card.classList.remove('active');
        if (card.dataset.character === characterId) {
            card.classList.add('active');
        }
    });
}

function updateChatHeader(characterId) {
    const characters = {
        'moses': {
            name: 'Moses',
            description: 'Biblical Prophet and Lawgiver',
            avatar: 'static/avatars/moses.svg'
        },
        'samsung_employee': {
            name: 'Samsung Employee',
            description: 'Tech Expert and Product Specialist',
            avatar: 'static/avatars/samsung.svg'
        },
        'jinx': {
            name: 'Jinx',
            description: 'Chaotic Genius from Arcane',
            avatar: 'static/avatars/jinx.svg'
        }
    };
    
    const character = characters[characterId];
    if (character) {
        document.getElementById('current-character-name').textContent = character.name;
        document.getElementById('current-character-desc').textContent = character.description;
        document.getElementById('current-avatar').src = character.avatar;
        document.getElementById('message-input').placeholder = `Message ${character.name}...`;
        document.getElementById('typing-character').textContent = character.name;
    }
}

function sendMessage() {
    const messageInput = document.getElementById('message-input');
    const message = messageInput.value.trim();
    
    if (!message || !ws || ws.readyState !== WebSocket.OPEN) return;
    
    // Display user message
    displayMessage(message, 'user');
    
    // Add to conversation history
    conversationHistory.push({
        role: 'user',
        content: message
    });
    
    // Show typing indicator
    showTypingIndicator();
    
    // Send via WebSocket
    ws.send(JSON.stringify({
        type: 'chat_message',
        text: message,
        history: conversationHistory.slice(-10), // Keep last 10 messages
        include_audio: voiceEnabled,
        timestamp: Date.now()
    }));
    
    // Clear input
    messageInput.value = '';
    autoResize(messageInput);
}

function displayMessage(content, sender, characterId = null) {
    const chatMessages = document.getElementById('chat-messages');
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${sender}`;
    
    let avatarSrc = '';
    if (sender === 'user') {
        avatarSrc = 'static/avatars/user.svg';
    } else {
        const avatars = {
            'moses': 'static/avatars/moses.svg',
            'samsung_employee': 'static/avatars/samsung.svg',
            'jinx': 'static/avatars/jinx.svg'
        };
        avatarSrc = avatars[characterId || currentCharacter];
    }
    
    messageDiv.innerHTML = `
        <div class="message-avatar">
            <img src="${avatarSrc}" alt="${sender}" onerror="this.src='static/avatars/default.svg'">
        </div>
        <div class="message-content">
            <p>${escapeHtml(content)}</p>
        </div>
    `;
    
    chatMessages.appendChild(messageDiv);
    scrollToBottom();
    
    // Add to conversation history for assistant messages
    if (sender === 'assistant') {
        conversationHistory.push({
            role: 'assistant',
            content: content
        });
    }
}

function showWelcomeMessage(characterId) {
    const welcomeMessages = {
        'moses': "Peace be with you, my child. I am Moses, prophet and lawgiver. My character-focused training allows me to embody the wisdom of the Almighty. How may I guide you in righteousness?",
        'samsung_employee': "Hello! I'm your Samsung technology expert, trained with character-focused learning to provide authentic product knowledge and enthusiasm. What amazing Galaxy features can I share with you today?",
        'jinx': "*spins around excitedly* Hey there! I'm Jinx - the real me, not some AI pretending! My character-focused training means pure chaotic genius with no boring assistant stuff. Ready for some explosive fun?"
    };
    
    const message = welcomeMessages[characterId];
    if (message) {
        displayMessage(message, 'assistant', characterId);
    }
}

function showTypingIndicator() {
    document.getElementById('typing-indicator').style.display = 'flex';
    scrollToBottom();
}

function hideTypingIndicator() {
    document.getElementById('typing-indicator').style.display = 'none';
}

function scrollToBottom() {
    const chatMessages = document.getElementById('chat-messages');
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

function playAudio(audioData) {
    if (!voiceEnabled || !audioData) return;
    
    try {
        const audio = new Audio(audioData);
        audio.play().catch(error => {
            console.error('Error playing audio:', error);
        });
    } catch (error) {
        console.error('Error creating audio:', error);
    }
}

function toggleVoice() {
    voiceEnabled = !voiceEnabled;
    const voiceIcon = document.getElementById('voice-icon');
    const voiceToggle = document.querySelector('.voice-toggle');
    
    if (voiceEnabled) {
        voiceIcon.className = 'fas fa-volume-up';
        voiceToggle.classList.add('active');
    } else {
        voiceIcon.className = 'fas fa-volume-mute';
        voiceToggle.classList.remove('active');
    }
    
    saveSettings();
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

function showResources() {
    const modal = document.getElementById('resources-modal');
    modal.classList.add('active');
}

function showSettings() {
    const modal = document.getElementById('settings-modal');
    modal.classList.add('active');
}

function closeModal(modalId) {
    const modal = document.getElementById(modalId);
    modal.classList.remove('active');
}

function updateVoiceSetting() {
    const voiceCheckbox = document.getElementById('voice-enabled');
    voiceEnabled = voiceCheckbox.checked;
    
    const voiceToggle = document.querySelector('.voice-toggle');
    const voiceIcon = document.getElementById('voice-icon');
    
    if (voiceEnabled) {
        voiceToggle.classList.add('active');
        voiceIcon.className = 'fas fa-volume-up';
    } else {
        voiceToggle.classList.remove('active');
        voiceIcon.className = 'fas fa-volume-mute';
    }
    
    saveSettings();
}

function changeTheme() {
    const themeSelect = document.getElementById('theme-select');
    const theme = themeSelect.value;
    
    document.documentElement.setAttribute('data-theme', theme);
    saveSettings();
}

function saveSettings() {
    const settings = {
        voiceEnabled: voiceEnabled,
        theme: document.documentElement.getAttribute('data-theme') || 'dark',
        responseSpeed: document.getElementById('response-speed')?.value || '0.7'
    };
    
    localStorage.setItem('roleplayChatSettings', JSON.stringify(settings));
}

function loadSettings() {
    try {
        const settings = JSON.parse(localStorage.getItem('roleplayChatSettings')) || {};
        
        // Load voice setting
        if (settings.voiceEnabled !== undefined) {
            voiceEnabled = settings.voiceEnabled;
            document.getElementById('voice-enabled').checked = voiceEnabled;
            updateVoiceSetting();
        }
        
        // Load theme
        if (settings.theme) {
            document.documentElement.setAttribute('data-theme', settings.theme);
            document.getElementById('theme-select').value = settings.theme;
        }
        
        // Load response speed
        if (settings.responseSpeed) {
            document.getElementById('response-speed').value = settings.responseSpeed;
        }
    } catch (error) {
        console.error('Error loading settings:', error);
    }
}

// Close modals when clicking outside
window.addEventListener('click', function(event) {
    if (event.target.classList.contains('modal')) {
        event.target.classList.remove('active');
    }
});

// Handle escape key to close modals
document.addEventListener('keydown', function(event) {
    if (event.key === 'Escape') {
        const activeModal = document.querySelector('.modal.active');
        if (activeModal) {
            activeModal.classList.remove('active');
        }
    }
});