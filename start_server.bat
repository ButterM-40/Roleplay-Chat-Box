# Deployment and Testing Scripts

# Start backend server
cd backend
python main.py

# In another terminal, you can test the API endpoints
# Test character list
curl http://localhost:8000/api/characters

# Test chat endpoint
curl -X POST http://localhost:8000/api/chat/moses \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello Moses", "conversation_history": []}'