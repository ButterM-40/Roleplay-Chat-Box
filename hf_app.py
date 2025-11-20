import gradio as gr
import os
import sys
import asyncio
import logging
from typing import List, Tuple, Optional

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add backend to path
backend_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'backend')
sys.path.insert(0, backend_path)

# Global character manager
character_manager = None
models_loaded = False

def initialize_models():
    """Initialize the character manager"""
    global character_manager, models_loaded
    
    if models_loaded:
        return "✅ Models already loaded!"
    
    try:
        from backend.models.character_manager import CharacterManager
        
        character_manager = CharacterManager()
        
        # Initialize synchronously 
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(character_manager.initialize())
        
        models_loaded = True
        logger.info("✅ Character models initialized successfully!")
        return "✅ Models loaded successfully!"
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize models: {e}")
        return f"❌ Failed to load models: {str(e)}"

def get_character_info():
    """Get character information for display"""
    return {
        "moses": {
            "name": "Moses",
            "description": "📚 Wise biblical figure offering guidance and wisdom",
            "avatar": "👨‍🏫"
        },
        "samsung_employee": {
            "name": "Samsung Employee", 
            "description": "💼 Professional tech support specialist",
            "avatar": "👨‍💼"
        },
        "jinx": {
            "name": "Jinx",
            "description": "🎭 Chaotic and energetic character from Arcane", 
            "avatar": "🔮"
        }
    }

def chat_with_character(message: str, character_id: str, history: List[Tuple[str, str]]) -> Tuple[List[Tuple[str, str]], str]:
    """Generate character response and update chat history"""
    global character_manager, models_loaded
    
    # Initialize if needed
    if not models_loaded:
        init_result = initialize_models()
        if "Failed" in init_result:
            return history + [(message, init_result)], ""
    
    if not message.strip():
        return history, ""
    
    try:
        if character_manager is None:
            return history + [(message, "❌ Character manager not initialized")], ""
        
        # Generate response
        response = character_manager.generate_response(
            character_id=character_id,
            user_input=message,
            max_length=512
        )
        
        # Update history
        new_history = history + [(message, response)]
        return new_history, ""
        
    except Exception as e:
        logger.error(f"Error generating response: {e}")
        error_response = f"❌ Error: {str(e)}"
        return history + [(message, error_response)], ""

def get_character_display_html(character_id: str) -> str:
    """Generate HTML for character display"""
    char_info = get_character_info()
    if character_id not in char_info:
        return "<div>Character not found</div>"
    
    info = char_info[character_id]
    return f"""
    <div style="text-align: center; padding: 20px; background: linear-gradient(45deg, #667eea 0%, #764ba2 100%); 
                color: white; border-radius: 15px; margin: 10px;">
        <div style="font-size: 4rem; margin-bottom: 10px;">{info['avatar']}</div>
        <h2 style="margin: 10px 0; color: white;">{info['name']}</h2>
        <p style="margin: 0; opacity: 0.9; font-size: 1.1rem;">{info['description']}</p>
    </div>
    """

def create_interface():
    """Create the main Gradio interface"""
    
    # Custom CSS
    custom_css = """
    .gradio-container {
        max-width: 1200px !important;
    }
    .character-display {
        min-height: 200px;
    }
    .chat-container {
        height: 500px;
    }
    """
    
    with gr.Blocks(
        title="🎭 Roleplay Chat Box",
        theme=gr.themes.Soft(primary_hue="purple"),
        css=custom_css
    ) as demo:
        
        gr.Markdown("# 🎭 Roleplay Chat Box")
        gr.Markdown("### Chat with AI characters, each with unique personalities!")
        
        with gr.Row():
            # Character selection column
            with gr.Column(scale=1):
                gr.Markdown("## 👥 Choose Character")
                
                character_dropdown = gr.Dropdown(
                    choices=[
                        ("👨‍🏫 Moses", "moses"),
                        ("👨‍💼 Samsung Employee", "samsung_employee"), 
                        ("🔮 Jinx", "jinx")
                    ],
                    value="moses",
                    label="Select Character",
                    interactive=True
                )
                
                # Character info display
                character_display = gr.HTML(
                    value=get_character_display_html("moses"),
                    elem_classes=["character-display"]
                )
                
                # Update character display when selection changes
                character_dropdown.change(
                    fn=get_character_display_html,
                    inputs=[character_dropdown],
                    outputs=[character_display]
                )
            
            # Chat column
            with gr.Column(scale=2):
                gr.Markdown("## 💬 Chat")
                
                chatbot = gr.Chatbot(
                    height=500,
                    show_label=False,
                    elem_classes=["chat-container"]
                )
                
                with gr.Row():
                    msg_input = gr.Textbox(
                        placeholder="Type your message here...",
                        show_label=False,
                        scale=4,
                        lines=2
                    )
                    
                    with gr.Column(scale=1):
                        send_btn = gr.Button("Send 📨", variant="primary")
                        clear_btn = gr.Button("Clear 🗑️")
        
        # Status section
        with gr.Row():
            status_display = gr.Textbox(
                value="Click 'Initialize Models' to start chatting!",
                label="Status",
                interactive=False
            )
            init_btn = gr.Button("Initialize Models 🚀", variant="secondary")
        
        # Event handlers
        def send_message(message, character_id, history):
            return chat_with_character(message, character_id, history)
        
        def clear_chat():
            return [], "Chat cleared!"
        
        def init_models_handler():
            return initialize_models()
        
        # Button clicks
        send_btn.click(
            fn=send_message,
            inputs=[msg_input, character_dropdown, chatbot],
            outputs=[chatbot, msg_input]
        )
        
        msg_input.submit(
            fn=send_message,
            inputs=[msg_input, character_dropdown, chatbot],
            outputs=[chatbot, msg_input]
        )
        
        clear_btn.click(
            fn=clear_chat,
            outputs=[chatbot, status_display]
        )
        
        init_btn.click(
            fn=init_models_handler,
            outputs=[status_display]
        )
        
        # Example interactions
        gr.Markdown("""
        ### 💡 Example Conversations
        - **Moses**: "What is the meaning of wisdom?"
        - **Samsung Employee**: "Tell me about the latest Samsung phones"
        - **Jinx**: "I need help with a creative project!"
        """)
    
    return demo

if __name__ == "__main__":
    # Create and launch the interface
    demo = create_interface()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True,
        share=False
    )