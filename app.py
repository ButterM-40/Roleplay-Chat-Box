import gradio as gr
import requests
import json
import time
import os
import sys
from typing import List, Tuple

# Add backend to path for imports
backend_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'backend')
sys.path.insert(0, backend_path)

from backend.models.character_manager import CharacterManager
from backend.config import settings

class RoleplayChatInterface:
    def __init__(self):
        """Initialize the Roleplay Chat Interface"""
        self.character_manager = None
        self.available_characters = ["moses", "samsung_employee", "jinx"]
        self.character_info = {
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
        
    async def initialize_models(self):
        """Initialize the character manager"""
        try:
            self.character_manager = CharacterManager()
            await self.character_manager.initialize()
            return "✅ Models loaded successfully!"
        except Exception as e:
            return f"❌ Failed to load models: {str(e)}"
    
    def initialize_models_sync(self):
        """Synchronous wrapper for model initialization"""
        import asyncio
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            result = loop.run_until_complete(self.initialize_models())
            loop.close()
            return result
        except Exception as e:
            return f"❌ Failed to load models: {str(e)}"
    
    def get_character_response(self, message: str, character_id: str, history: List[Tuple[str, str]]) -> Tuple[List[Tuple[str, str]], str]:
        """Generate character response and update chat history"""
        if not self.character_manager:
            return history + [(message, "⚠️ Models are still loading. Please try again in a moment...")], ""
        
        if not message.strip():
            return history, ""
        
        try:
            # Convert Gradio history to conversation format
            conversation_history = []
            for user_msg, assistant_msg in history[-3:]:  # Last 3 exchanges for context
                conversation_history.append({"role": "user", "content": user_msg})
                if assistant_msg:
                    conversation_history.append({"role": "assistant", "content": assistant_msg})
            
            # Generate response using character manager (correct method signature)
            response = self.character_manager.generate_response(
                character_id=character_id,
                user_message=message,
                conversation_history=conversation_history
            )
            
            # Update chat history
            new_history = history + [(message, response)]
            return new_history, ""
            
        except Exception as e:
            error_response = f"❌ Error generating response: {str(e)}"
            new_history = history + [(message, error_response)]
            return new_history, ""
    
    def get_character_options(self):
        """Get character dropdown options"""
        options = []
        for char_id in self.available_characters:
            info = self.character_info[char_id]
            label = f"{info['avatar']} {info['name']}"
            options.append((label, char_id))
        return options
    
    def create_interface(self):
        """Create the Gradio interface"""
        
        with gr.Blocks(
            title="🎭 Roleplay Chat Box",
            theme=gr.themes.Soft(primary_hue="purple"),
            css="""
            .character-info {
                background: linear-gradient(45deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 1rem;
                border-radius: 10px;
                margin-bottom: 1rem;
            }
            .chat-container {
                max-height: 500px;
                overflow-y: auto;
            }
            """) as iface:
            
            gr.Markdown("# 🎭 Roleplay Chat Box")
            gr.Markdown("Chat with different AI characters, each with unique personalities and expertise!")
            
            with gr.Row():
                with gr.Column(scale=1):
                    # Character Selection
                    gr.Markdown("## 👥 Choose Your Character")
                    character_dropdown = gr.Dropdown(
                        choices=self.get_character_options(),
                        value="moses",
                        label="Select Character",
                        interactive=True
                    )
                    
                    # Character Info Display
                    character_info_display = gr.HTML(
                        value=self._get_character_info_html("moses"),
                        elem_classes=["character-info"]
                    )
                    
                    # Update character info when dropdown changes
                    def update_character_info(character_id):
                        return self._get_character_info_html(character_id)
                    
                    character_dropdown.change(
                        fn=update_character_info,
                        inputs=[character_dropdown],
                        outputs=[character_info_display]
                    )
                
                with gr.Column(scale=2):
                    # Chat Interface
                    gr.Markdown("## 💬 Chat")
                    chatbot = gr.Chatbot(
                        height=400,
                        show_label=False,
                        container=True,
                        elem_classes=["chat-container"]
                    )
                    
                    with gr.Row():
                        msg_input = gr.Textbox(
                            placeholder="Type your message here...",
                            show_label=False,
                            scale=4,
                            lines=1
                        )
                        send_btn = gr.Button("Send 📨", scale=1, variant="primary")
                        clear_btn = gr.Button("Clear 🗑️", scale=1)
            
            # Initialize models on startup
            gr.Markdown("### 🔄 Status")
            status_display = gr.Textbox(value="Loading models...", label="System Status", interactive=False)
            
            # Chat functionality
            def respond_and_clear(message, character_id, history):
                new_history, _ = self.get_character_response(message, character_id, history)
                return new_history, ""
            
            # Send message on button click or Enter
            send_btn.click(
                fn=respond_and_clear,
                inputs=[msg_input, character_dropdown, chatbot],
                outputs=[chatbot, msg_input]
            )
            
            msg_input.submit(
                fn=respond_and_clear,
                inputs=[msg_input, character_dropdown, chatbot], 
                outputs=[chatbot, msg_input]
            )
            
            # Clear chat
            clear_btn.click(
                fn=lambda: ([], "Chat cleared!"),
                outputs=[chatbot, status_display]
            )
            
            # Initialize models when interface loads
            def init_models():
                if not self.character_manager:
                    return self.initialize_models_sync()
                else:
                    return "✅ Models already loaded!"
            
            iface.load(
                fn=init_models,
                outputs=[status_display]
            )
        
        return iface
    
    def _get_character_info_html(self, character_id: str) -> str:
        """Generate HTML for character information"""
        if character_id not in self.character_info:
            return "<div>Character not found</div>"
        
        info = self.character_info[character_id]
        return f"""
        <div style="text-align: center;">
            <div style="font-size: 3rem; margin-bottom: 0.5rem;">{info['avatar']}</div>
            <h3 style="margin: 0.5rem 0; color: white;">{info['name']}</h3>
            <p style="margin: 0; opacity: 0.9;">{info['description']}</p>
        </div>
        """

# Create and launch the interface
def create_demo():
    chat_interface = RoleplayChatInterface()
    return chat_interface.create_interface()

# For Hugging Face Spaces
if __name__ == "__main__":
    demo = create_demo()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True
    )