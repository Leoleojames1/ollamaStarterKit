import sys
import os
import re
from PyQt6.QtWidgets import (QApplication, QMainWindow, QTextEdit, 
                            QLineEdit, QPushButton, QVBoxLayout, QWidget,
                            QSplitter, QLabel, QHBoxLayout, QComboBox,
                            QMessageBox, QTextBrowser, QCheckBox, QFileDialog)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt6.QtGui import QFont, QColor, QPalette, QIcon, QFontDatabase
import ollama

# Function to preprocess text for better display
def preprocess_text(text):
    # Fix spacing issues for common punctuation
    text = re.sub(r'([.,!?:;])(\w)', r'\1 \2', text)
    
    # Fix capitalization after sentence-ending punctuation
    text = re.sub(r'([.!?])\s*([a-z])', lambda m: m.group(1) + ' ' + m.group(2).upper(), text)
    
    # Add spacing between camelCase or PascalCase words
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
    
    return text

# Thread for handling Ollama API calls without freezing UI
class OllamaThread(QThread):
    response_signal = pyqtSignal(str)
    error_signal = pyqtSignal(str)
    complete_signal = pyqtSignal()
    start_signal = pyqtSignal()  # Signal to indicate that processing has started
    
    def __init__(self, prompt, model, parent=None):
        super().__init__(parent)
        self.prompt = prompt
        self.model = model
        self.last_char = " "  # Keep track of the last character to handle spacing
        
    def run(self):
        try:
            # Signal that processing has started
            self.start_signal.emit()
            
            stream = ollama.chat(
                model=self.model,
                messages=[{"role": "user", "content": self.prompt}],
                stream=True,
            )
            
            for chunk in stream:
                content = ""
                # Handle different response formats
                if hasattr(chunk, "message") and hasattr(chunk.message, "content"):
                    content = chunk.message.content
                elif isinstance(chunk, dict):
                    content = chunk.get('message', {}).get('content', '')
                
                if content:
                    # Fix spacing issues - add spaces if needed
                    if (self.last_char != " " and content and content[0].isalnum() and 
                        self.last_char.isalnum()):
                        content = " " + content
                    
                    # Update last character
                    if content:
                        self.last_char = content[-1]
                        
                    # Process the content for better readability
                    processed_content = preprocess_text(content)
                    
                    self.response_signal.emit(processed_content)
            
            # Signal that the response is complete
            self.complete_signal.emit()
                
        except Exception as e:
            self.error_signal.emit(f"\nError: {str(e)}")
            self.complete_signal.emit()

# Thread for fetching available models
class ModelListThread(QThread):
    models_signal = pyqtSignal(list)
    error_signal = pyqtSignal(str)
    
    def run(self):
        try:
            # Get the list of models from ollama
            result = ollama.list()
            
            # Handle different response formats
            model_names = []
            
            # Check if result has 'models' attribute (newer API format)
            if hasattr(result, 'models'):
                models = result.models
                for model in models:
                    if hasattr(model, 'model'):
                        model_names.append(model.model)
            # Legacy API format check
            elif isinstance(result, dict) and 'models' in result:
                models = result['models']
                model_names = [model.get('name', '') for model in models]
            else:
                self.error_signal.emit("Unrecognized response format from ollama.list()")
                return
                
            if model_names:
                self.models_signal.emit(model_names)
            else:
                self.error_signal.emit("No models found in the response")
        except Exception as e:
            self.error_signal.emit(str(e))

# Function to format code blocks in HTML
def format_code_blocks(text):
    def replace_code_block(match):
        language = match.group(1) or ''
        code = match.group(2)
        formatted = f'''
        <pre style="background-color: #2b2b2b; color: #a9b7c6; padding: 10px; border-radius: 5px; overflow-x: auto;">
            <code class="language-{language}" style="color: #a9b7c6;">{code}</code>
        </pre>
        '''
        return formatted
    
    # Find code blocks with language specification: ```python\ncode\n```
    pattern = r'```(\w+)?\n([\s\S]+?)\n```'
    return re.sub(pattern, replace_code_block, text)

class OllamaChatApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.active_threads = []  # Keep track of active threads
        self.is_dark_mode = False  # Default to light mode
        self.current_response = ""  # Track current response text
        self.message_counter = 0   # Keep track of message IDs
        self.messages = []        # Store messages for display
        self.is_typing = False    # Typing indicator state
        self.typing_dots = 0      # For animated dots
        self.typing_timer = QTimer()  # Timer for typing animation
        self.typing_timer.timeout.connect(self.update_typing_indicator)
        self.setup_fonts()
        self.init_ui()
        self.apply_theme()
        self.fetch_models()
        
    def setup_fonts(self):
        # Load and set up Arial font for better display
        font_id = QFontDatabase.addApplicationFont("arial.ttf")
        if font_id == -1:
            print("Warning: Arial font not found. Using default font.")
        # If Arial is not available, fall back to a system sans-serif font
        self.app_font = QFont("Arial", 14)
        self.app_font.setStyleHint(QFont.StyleHint.SansSerif)
        self.heading_font = QFont("Arial", 20)
        self.heading_font.setStyleHint(QFont.StyleHint.SansSerif)
        self.heading_font.setBold(True)
        
    def init_ui(self):
        # Main widget and layout
        main_widget = QWidget()
        main_layout = QVBoxLayout(main_widget)
        main_layout.setSpacing(15)
        
        # Header with logo and title
        header_layout = QHBoxLayout()
        
        logo_label = QLabel("🦙")  # Glass ball emoji as logo
        logo_label.setFont(QFont("Arial", 24))
        
        header = QLabel("Ollama 101")
        header.setFont(self.heading_font)
        header.setAlignment(Qt.AlignmentFlag.AlignLeft)
        
        # Theme toggle
        self.theme_toggle = QCheckBox("Dark Mode")
        self.theme_toggle.setFont(QFont("Arial", 12))
        self.theme_toggle.stateChanged.connect(self.toggle_theme)
        
        header_layout.addWidget(logo_label)
        header_layout.addWidget(header)
        header_layout.addStretch()
        header_layout.addWidget(self.theme_toggle)
        
        main_layout.addLayout(header_layout)
        
        # Model selection layout
        model_layout = QHBoxLayout()
        model_label = QLabel("Select Model:")
        model_label.setFont(self.app_font)
        
        self.model_combo = QComboBox()
        self.model_combo.setFont(self.app_font)
        self.model_combo.setMinimumHeight(40)
        self.model_combo.addItem("Loading models...")
        
        refresh_button = QPushButton("🔄")
        refresh_button.setFixedWidth(50)
        refresh_button.setFixedHeight(40)
        refresh_button.clicked.connect(self.fetch_models)
        
        model_layout.addWidget(model_label)
        model_layout.addWidget(self.model_combo, 3)
        model_layout.addWidget(refresh_button)
        
        main_layout.addLayout(model_layout)
        
        # Chat display
        self.chat_display = QTextBrowser()
        self.chat_display.setOpenExternalLinks(False)
        self.chat_display.setReadOnly(True)
        self.chat_display.setFont(self.app_font)
        self.chat_display.setMinimumHeight(400)
        
        main_layout.addWidget(self.chat_display, 4)
        
        # Typing indicator layout
        typing_layout = QHBoxLayout()
        self.typing_indicator = QLabel("")
        self.typing_indicator.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        self.typing_indicator.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.typing_indicator.setVisible(False)  # Hidden by default
        typing_layout.addStretch()
        typing_layout.addWidget(self.typing_indicator)
        typing_layout.addStretch()
        
        main_layout.addLayout(typing_layout)
        
        # Input area
        input_layout = QHBoxLayout()
        self.prompt_input = QLineEdit()
        self.prompt_input.setPlaceholderText("Enter your prompt here... (Use / for commands)")
        self.prompt_input.setMinimumHeight(50)
        self.prompt_input.setFont(self.app_font)
        self.prompt_input.returnPressed.connect(self.send_prompt)
        
        self.send_button = QPushButton("Send")
        self.send_button.setMinimumHeight(50)
        self.send_button.setFont(self.app_font)
        self.send_button.clicked.connect(self.send_prompt)
        
        self.upload_button = QPushButton("Upload File")
        self.upload_button.setMinimumHeight(50)
        self.upload_button.setFont(self.app_font)
        self.upload_button.clicked.connect(self.upload_file)
        
        input_layout.addWidget(self.prompt_input, 4)
        input_layout.addWidget(self.send_button, 1)
        input_layout.addWidget(self.upload_button, 1)
        
        main_layout.addLayout(input_layout, 1)
        
        # Set main widget
        self.setCentralWidget(main_widget)
        
        # Window settings
        self.setWindowTitle("Ollama Glass Chat")
        self.setGeometry(300, 300, 900, 700)
        self.show()
    
    def upload_file(self):
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(self, "Upload File", "", "All Files (*);;Text Files (*.txt);;Python Files (*.py);;Markdown Files (*.md)", options=options)
        if file_path:
            try:
                with open(file_path, 'r') as file:
                    file_content = file.read()
                    self.add_user_message(f"Uploaded file content:\n\n```\n{file_content}\n```")
            except Exception as e:
                self.add_system_message(f"Error reading file: {str(e)}")
    
    def update_typing_indicator(self):
        """Update the typing indicator animation"""
        if self.is_typing:
            # Update the spinner animation
            spinner_chars = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
            spinner_index = int((self.typing_dots / 2) % len(spinner_chars))
            spinner = spinner_chars[spinner_index]
            
            # Update the dots animation
            dots = "." * (self.typing_dots % 4)
            spaces = " " * (3 - (self.typing_dots % 4))
            
            # Set the indicator text with spinner and dots
            if self.is_dark_mode:
                self.typing_indicator.setText(f"<span style='color: #aaaaaa;'>{spinner} Predicting{dots}{spaces}</span>")
            else:
                self.typing_indicator.setText(f"<span style='color: #555555;'>{spinner} Predicting{dots}{spaces}</span>")
            
            self.typing_dots += 1
    
    def show_typing_indicator(self):
        """Show the typing indicator and start animation"""
        self.is_typing = True
        self.typing_indicator.setVisible(True)
        self.typing_dots = 0
        self.typing_timer.start(150)  # Update every 150ms for smooth animation
    
    def hide_typing_indicator(self):
        """Hide the typing indicator and stop animation"""
        self.is_typing = False
        self.typing_indicator.setVisible(False)
        self.typing_timer.stop()
        
    def apply_theme(self):
        # Apply theme based on current mode
        if self.is_dark_mode:
            # Dark theme with glass effect
            self.setStyleSheet("""
                QMainWindow, QWidget {
                    background-color: rgba(18, 18, 18, 0.95);
                    color: #ffffff;
                    font-family: 'Arial', sans-serif;
                }
                QTextEdit, QLineEdit, QComboBox, QTextBrowser {
                    background-color: rgba(30, 30, 30, 0.8);
                    color: #ffffff;
                    border: 1px solid #555555;
                    border-radius: 8px;
                    padding: 8px;
                    font-size: 14px;
                    font-family: 'Arial', sans-serif;
                }
                QPushButton {
                    background-color: rgba(60, 60, 60, 0.9);
                    color: #ffffff;
                    border: 1px solid #666666;
                    border-radius: 8px;
                    padding: 10px 18px;
                    font-weight: bold;
                    font-size: 14px;
                }
                QPushButton:hover {
                    background-color: rgba(80, 80, 80, 0.9);
                    border: 1px solid #888888;
                }
                QPushButton:pressed {
                    background-color: rgba(40, 40, 40, 0.9);
                }
                QLabel {
                    color: #ffffff;
                    font-weight: bold;
                    font-size: 16px;
                }
                QComboBox {
                    padding: 10px;
                    font-size: 14px;
                    background-color: rgba(30, 30, 30, 0.8);
                    selection-background-color: rgba(60, 60, 60, 0.9);
                }
                QComboBox::drop-down {
                    border: 0px;
                }
                QComboBox::down-arrow {
                    width: 14px;
                    height: 14px;
                }
                QScrollBar:vertical {
                    background-color: rgba(30, 30, 30, 0.5);
                    width: 14px;
                    margin: 0px;
                }
                QScrollBar::handle:vertical {
                    background-color: rgba(80, 80, 80, 0.8);
                    min-height: 20px;
                    border-radius: 7px;
                }
                QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                    height: 0px;
                }
                QCheckBox {
                    color: #ffffff;
                    font-size: 14px;
                }
                QCheckBox::indicator {
                    width: 18px;
                    height: 18px;
                    border-radius: 4px;
                    border: 1px solid #666666;
                    background-color: rgba(30, 30, 30, 0.8);
                }
                QCheckBox::indicator:checked {
                    background-color: rgba(100, 100, 100, 0.8);
                }
            """)
        else:
            # Light theme with glass effect
            self.setStyleSheet("""
                QMainWindow, QWidget {
                    background-color: rgba(240, 240, 245, 0.95);
                    color: #000000;
                    font-family: 'Arial', sans-serif;
                }
                QTextEdit, QLineEdit, QComboBox, QTextBrowser {
                    background-color: rgba(255, 255, 255, 0.8);
                    color: #000000;
                    border: 1px solid #aaaaaa;
                    border-radius: 8px;
                    padding: 8px;
                    font-size: 14px;
                    font-family: 'Arial', sans-serif;
                }
                QPushButton {
                    background-color: rgba(220, 220, 220, 0.9);
                    color: #000000;
                    border: 1px solid #cccccc;
                    border-radius: 8px;
                    padding: 10px 18px;
                    font-weight: bold;
                    font-size: 14px;
                }
                QPushButton:hover {
                    background-color: rgba(200, 200, 200, 0.9);
                    border: 1px solid #bbbbbb;
                }
                QPushButton:pressed {
                    background-color: rgba(180, 180, 180, 0.9);
                }
                QLabel {
                    color: #000000;
                    font-weight: bold;
                    font-size: 16px;
                }
                QComboBox {
                    padding: 10px;
                    font-size: 14px;
                    background-color: rgba(255, 255, 255, 0.8);
                    selection-background-color: rgba(220, 220, 220, 0.9);
                }
                QComboBox::drop-down {
                    border: 0px;
                }
                QComboBox::down-arrow {
                    width: 14px;
                    height: 14px;
                }
                QScrollBar:vertical {
                    background-color: rgba(230, 230, 240, 0.5);
                    width: 14px;
                    margin: 0px;
                }
                QScrollBar::handle:vertical {
                    background-color: rgba(180, 180, 200, 0.8);
                    min-height: 20px;
                    border-radius: 7px;
                }
                QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                    height: 0px;
                }
                QCheckBox {
                    color: #000000;
                    font-size: 14px;
                }
                QCheckBox::indicator {
                    width: 18px;
                    height: 18px;
                    border-radius: 4px;
                    border: 1px solid #aaaaaa;
                    background-color: rgba(255, 255, 255,.8);
                }
                QCheckBox::indicator:checked {
                    background-color: rgba(180, 180, 180, 0.8);
                }
            """)
        
        # Refresh the display with the new theme
        self.refresh_chat_display()
    
    def toggle_theme(self, state):
        self.is_dark_mode = bool(state)
        self.apply_theme()
    
    def clean_thread(self, thread):
        # Remove the thread from our active threads list when it's done
        if thread in self.active_threads:
            self.active_threads.remove(thread)
    
    def closeEvent(self, event):
        # Properly clean up threads when the application closes
        for thread in self.active_threads:
            thread.wait()
        event.accept()
    
    def fetch_models(self):
        self.model_combo.clear()
        self.model_combo.addItem("Loading models...")
        self.model_combo.setEnabled(False)
        self.send_button.setEnabled(False)
        
        self.model_thread = ModelListThread()
        self.model_thread.models_signal.connect(self.update_model_list)
        self.model_thread.error_signal.connect(self.show_model_error)
        self.model_thread.finished.connect(lambda: self.clean_thread(self.model_thread))
        self.active_threads.append(self.model_thread)
        self.model_thread.start()
    
    def update_model_list(self, models):
        self.model_combo.clear()
        if models:
            self.model_combo.addItems(models)
            self.model_combo.setCurrentIndex(0)
            
            # Add system message about available models
            system_message = {
                "id": f"system-{self.message_counter}",
                "type": "system",
                "content": "<b>Available models:</b><ul>"
            }
            self.message_counter += 1
            
            for model_name in models:
                system_message["content"] += f"<li><code>{model_name}</code></li>"
            system_message["content"] += "</ul>"
            
            self.messages.append(system_message)
            self.refresh_chat_display()
        else:
            self.model_combo.addItem("No models found")
            
            # Add system message about no models
            system_message = {
                "id": f"system-{self.message_counter}",
                "type": "system",
                "content": "No models found. Please pull a model using 'ollama pull &lt;model&gt;' command."
            }
            self.message_counter += 1
            self.messages.append(system_message)
            self.refresh_chat_display()
        
        self.model_combo.setEnabled(True)
        self.send_button.setEnabled(True)
    
    def show_model_error(self, error_msg):
        self.model_combo.clear()
        self.model_combo.addItem("Error loading models")
        
        # Add system message about error
        system_message = {
            "id": f"system-{self.message_counter}",
            "type": "system",
            "content": f"<b>Error loading models:</b> {error_msg}<br>Make sure Ollama is installed and running. Try 'ollama serve' in a terminal."
        }
        self.message_counter += 1
        self.messages.append(system_message)
        self.refresh_chat_display()
        
        self.model_combo.setEnabled(True)
        self.send_button.setEnabled(True)
    
    def add_system_message(self, content):
        system_message = {
            "id": f"system-{self.message_counter}",
            "type": "system",
            "content": content
        }
        self.message_counter += 1
        self.messages.append(system_message)
        self.refresh_chat_display()
    
    def add_user_message(self, content):
        user_message = {
            "id": f"user-{self.message_counter}",
            "type": "user",
            "content": content
        }
        self.message_counter += 1
        self.messages.append(user_message)
        self.refresh_chat_display()
    
    def add_assistant_message(self, model_name):
        # Create a new assistant message
        self.current_response = ""
        assistant_message = {
            "id": f"assistant-{self.message_counter}",
            "type": "assistant",
            "model": model_name,
            "content": ""
        }
        self.message_counter += 1
        self.messages.append(assistant_message)
        self.refresh_chat_display()
        return assistant_message["id"]
    
    def update_assistant_message(self, message_id, content):
        # Find the message by ID and update its content
        for message in self.messages:
            if message["id"] == message_id:
                message["content"] += content
                self.refresh_chat_display()
                break
    
    def refresh_chat_display(self):
        # Clear display and rebuild it from our message list
        self.chat_display.clear()
        
        # Add welcome message if this is first load
        if not self.messages:
            welcome_html = """
            <div style="padding: 15px; margin: 10px 0; border-radius: 8px; background-color: rgba(150, 150, 150, 0.1);">
                <h2>Welcome to Ollama Glass Chat!</h2>
                <p>Select a model and start chatting. You can use Markdown in your messages for formatting.</p>
                <p><b>Available commands:</b></p>
                <ul>
                    <li><code>/help</code> - Show this help message</li>
                    <li><code>/list</code> - Refresh the model list</li>
                    <li><code>/swap &lt;model&gt;</code> - Switch to a different model</li>
                    <li><code>/clear</code> - Clear the chat history</li>
                </ul>
            </div>
            """
            self.chat_display.setHtml(welcome_html)
            return
        
        # Build HTML for all messages
        chat_html = ""
        for message in self.messages:
            if message["type"] == "system":
                # System message style - subtle background
                bg_color = "rgba(150, 150, 150, 0.1)" 
                text_color = "#555555" if not self.is_dark_mode else "#cccccc"
                
                chat_html += f"""
                <div style="padding: 15px; margin: 10px 0; border-radius: 8px; background-color: {bg_color}; color: {text_color};">
                    {message["content"]}
                </div>
                """
            
            elif message["type"] == "user":
                # User message style - light or dark based on theme
                bg_color = "rgba(220, 220, 220, 0.2)" if not self.is_dark_mode else "rgba(60, 60, 60, 0.3)"
                border_color = "#cccccc" if not self.is_dark_mode else "#555555"
                
                chat_html += f"""
                <div style="padding: 15px; margin: 10px 0; border-radius: 8px; 
                            background-color: {bg_color}; border-left: 3px solid {border_color};">
                    <div style="font-weight: bold; margin-bottom: 5px;">You:</div>
                    <div>{message["content"]}</div>
                </div>
                """
            
            elif message["type"] == "assistant":
                # Assistant message style - with model name
                bg_color = "rgba(240, 240, 240, 0.3)" if not self.is_dark_mode else "rgba(40, 40, 40, 0.3)"
                border_color = "#aaaaaa" if not self.is_dark_mode else "#666666"
                
                # Format code blocks
                formatted_content = format_code_blocks(message["content"])
                
                chat_html += f"""
                <div style="padding: 15px; margin: 10px 0; border-radius: 8px; 
                            background-color: {bg_color}; border-left: 3px solid {border_color};">
                    <div style="font-weight: bold; margin-bottom: 5px;">Ollama ({message["model"]}):</div>
                    <div>{formatted_content}</div>
                </div>
                """
        
        # Update display
        self.chat_display.setHtml(chat_html)
        
        # Auto scroll to bottom
        self.chat_display.verticalScrollBar().setValue(
            self.chat_display.verticalScrollBar().maximum()
        )
    
    def send_prompt(self):
        prompt = self.prompt_input.text().strip()
        if not prompt:
            return
        
        # Clear input field
        self.prompt_input.clear()
        
        # Disable input while processing
        self.prompt_input.setEnabled(False)
        self.send_button.setEnabled(False)
        
        # Handle commands
        if prompt.startswith('/'):
            if prompt.startswith('/help'):
                # Show help message
                help_html = """
                <h3>Available Commands:</h3>
                <ul>
                    <li><code>/help</code> - Show this help message</li>
                    <li><code>/list</code> - Refresh the model list</li>
                    <li><code>/swap &lt;model&gt;</code> - Switch to a different model</li>
                    <li><code>/clear</code> - Clear the chat history</li>
                </ul>
                """
                self.add_system_message(help_html)
                return
                
            elif prompt.startswith('/list') or prompt.startswith('/ollama list'):
                # Refresh model list
                self.add_system_message("<p>Refreshing model list...</p>")
                self.fetch_models()
                return
                
            elif prompt == '/clear':
                # Clear chat history
                self.messages = []
                welcome_html = """
                <p>Chat history cleared.</p>
                <p><b>Available commands:</b> /help, /list, /swap &lt;model&gt;, /clear</p>
                """
                self.add_system_message(welcome_html)
                return
                
            elif prompt.startswith('/swap '):
                # Switch to a different model
                # First, wait for any ongoing requests to complete
                for thread in self.active_threads:
                    if isinstance(thread, OllamaThread):
                        thread.wait()
                
                model_name = prompt.split('/swap ', 1)[1].strip()
                
                # Try exact match first
                index = self.model_combo.findText(model_name)
                
                # If not found, try partial match
                if index == -1:
                    for i in range(self.model_combo.count()):
                        item_text = self.model_combo.itemText(i)
                        if model_name.lower() in item_text.lower():
                            index = i
                            break
                
                if index != -1:
                    self.model_combo.setCurrentIndex(index)
                    current_model = self.model_combo.currentText()
                    self.add_system_message(f"<p>Switched to model: <b>{current_model}</b></p>")
                else:
                    self.add_system_message(f"<p>Model '{model_name}' not found. Use /list to see available models.</p>")
                return
                
            # If it's an unknown command, treat it as a regular message
        
        # Regular message processing (not a command)
        selected_model = self.model_combo.currentText()
        if selected_model in ["Loading models...", "No models found", "Error loading models"]:
            QMessageBox.warning(self, "Model Selection", "Please select a valid model first.")
            return
            
        # Display user message
        self.add_user_message(prompt)
        
        # Create assistant message and get its ID for updating
        message_id = self.add_assistant_message(selected_model)
        
        # Start thread for API call
        self.ollama_thread = OllamaThread(prompt, selected_model)
        
        # Connect signals to update the specific message
        self.ollama_thread.start_signal.connect(self.show_typing_indicator)
        self.ollama_thread.response_signal.connect(lambda chunk: self.update_assistant_message(message_id, chunk))
        self.ollama_thread.error_signal.connect(lambda error: self.update_assistant_message(message_id, f"<span style='color: red;'>{error}</span>"))
        
        # Clean up when done
        self.ollama_thread.complete_signal.connect(self.hide_typing_indicator)
        self.ollama_thread.finished.connect(lambda: self.clean_thread(self.ollama_thread))
        self.ollama_thread.finished.connect(lambda: self.prompt_input.setEnabled(True))
        self.ollama_thread.finished.connect(lambda: self.send_button.setEnabled(True))
        
        self.active_threads.append(self.ollama_thread)
        self.ollama_thread.start()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = OllamaChatApp()
    sys.exit(app.exec())