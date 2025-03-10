# Add to top of file
import pandas as pd
import os
from datetime import datetime
import numpy as np
from pathlib import Path
import sys
import re
from PyQt6.QtWidgets import (QApplication, QMainWindow, QTextEdit, 
                            QLineEdit, QPushButton, QVBoxLayout, QWidget,
                            QSplitter, QLabel, QHBoxLayout, QComboBox,
                            QMessageBox, QTextBrowser, QCheckBox, QFileDialog,
                            QMenu)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer, QUrl
from PyQt6.QtGui import (QFont, QColor, 
                        QPalette, QIcon, 
                        QFontDatabase, QDesktopServices, 
                        QAction)
import ollama

# Default save location is user's home directory /.ollama_chat
DEFAULT_SAVE_DIR = os.path.join(str(Path.home()), '.ollama_chat')
SAVE_DIR = os.getenv('OLLAMA_CHAT_SAVE_DIR', DEFAULT_SAVE_DIR)

# Create save directory if it doesn't exist
os.makedirs(SAVE_DIR, exist_ok=True)

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
        code = match.group(2).strip()
        
        # Regular code block with syntax highlighting
        highlighted_code = syntax_highlight(code, language)
        
        return f'''
        <div class="code-block">
            <div class="code-header">
                <span>{language}</span>
            </div>
            <pre><code>{highlighted_code}</code></pre>
        </div>
        '''

    # First process code blocks
    text = re.sub(r'```(\w+)?\n([\s\S]+?)\n```', replace_code_block, text)
    
    # Then process other markdown elements
    text = process_markdown(text)
    
    return text

def process_markdown(text):
    """Process non-code markdown elements"""
    patterns = {
        # Headers
        r'^(#{1,6})\s(.+)$': lambda m: f'<h{len(m.group(1))} style="color: #BD93F9">{m.group(2)}</h{len(m.group(1))}>',
        
        # Bold
        r'\*\*(.+?)\*\*': r'<strong style="color: #FFB86C">\1</strong>',
        
        # Italic
        r'\*(.+?)\*': r'<em style="color: #F1FA8C">\1</em>',
        
        # Inline code
        r'`([^`]+)`': r'<code style="background: #44475A; padding: 0.2em 0.4em; border-radius: 3px">\1</code>',
        
        # Lists
        r'^\s*[-*]\s(.+)$': r'<li style="color: #F8F8F2">• \1</li>',
        
        # Links
        r'\[([^\]]+)\]\(([^\)]+)\)': r'<a href="\2" style="color: #8BE9FD">\1</a>'
    }
    
    for pattern, replacement in patterns.items():
        text = re.sub(pattern, replacement, text, flags=re.MULTILINE)
    
    return text

def syntax_highlight(code, language):
    """Enhanced syntax highlighter with more language support"""
    # Add language-specific syntax highlighting rules
    languages = {
        'python': {
            'keywords': ['def', 'class', 'import', 'from', 'return', 'if', 'else', 'elif', 
                        'for', 'while', 'try', 'except', 'with', 'as', 'lambda', 'yield'],
            'builtins': ['print', 'len', 'str', 'int', 'float', 'list', 'dict', 'set', 'tuple'],
            'string_color': '#F1FA8C',
            'keyword_color': '#FF79C6',
            'builtin_color': '#8BE9FD',
            'number_color': '#BD93F9',
            'comment_color': '#6272A4',
        },
        'javascript': {
            'keywords': ['function', 'const', 'let', 'var', 'return', 'if', 'else', 
                        'for', 'while', 'try', 'catch', 'class', 'extends'],
            'builtins': ['console', 'document', 'window', 'Array', 'Object', 'String'],
            'string_color': '#F1FA8C',
            'keyword_color': '#FF79C6',
            'builtin_color': '#8BE9FD',
            'number_color': '#BD93F9',
            'comment_color': '#6272A4',
        }
    }
    
    lang_rules = languages.get(language.lower(), languages['python'])
    
    # Apply syntax highlighting
    highlighted = code
    
    # Highlight strings
    highlighted = re.sub(
        r'(".*?"|\'.*?\')', 
        f'<span style="color: {lang_rules["string_color"]}">\\1</span>',
        highlighted
    )
    
    # Highlight comments
    highlighted = re.sub(
        r'(#.*?$|//.*?$)', 
        f'<span style="color: {lang_rules["comment_color"]}">\\1</span>',
        highlighted,
        flags=re.MULTILINE
    )
    
    # Highlight keywords
    for keyword in lang_rules['keywords']:
        highlighted = re.sub(
            f'\\b({keyword})\\b',
            f'<span style="color: {lang_rules["keyword_color"]}">\\1</span>',
            highlighted
        )
    
    # Highlight builtins
    for builtin in lang_rules['builtins']:
        highlighted = re.sub(
            f'\\b({builtin})\\b',
            f'<span style="color: {lang_rules["builtin_color"]}">\\1</span>',
            highlighted
        )
    
    # Highlight numbers
    highlighted = re.sub(
        r'\b(\d+)\b',
        f'<span style="color: {lang_rules["number_color"]}">\\1</span>',
        highlighted
    )
    
    return highlighted

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
        # Remove the font loading attempt and just use system fonts
        self.app_font = QFont("Arial", 18)
        self.app_font.setStyleHint(QFont.StyleHint.SansSerif)
        self.heading_font = QFont("Arial", 22)
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

        # Create stop button
        self.stop_button = QPushButton("□")  # Square symbol
        self.stop_button.setFixedSize(24, 24)  # Make it square
        self.stop_button.setStyleSheet("""
            QPushButton {
                background-color: transparent;
                border: 2px solid #ff0000;
                border-radius: 4px;
                color: #ff0000;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: rgba(255, 0, 0, 0.1);
            }
            QPushButton:pressed {
                background-color: rgba(255, 0, 0, 0.2);
            }
        """)
        self.stop_button.clicked.connect(self.stop_generation)
        self.stop_button.setVisible(False)  # Hidden by default

        typing_layout.addStretch()
        typing_layout.addWidget(self.typing_indicator)
        typing_layout.addWidget(self.stop_button)
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
        self.model_combo.currentIndexChanged.connect(self.on_model_changed)
    
    def upload_file(self):
        """Upload and handle files, preserving existing prompt text"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Upload File",
            "",
            "All Files (*);;Images (*.png *.jpg *.jpeg);;Text Files (*.txt);;Python Files (*.py)"
        )
        
        if file_path:
            try:
                # Get current prompt text
                current_prompt = self.prompt_input.text().strip()
                
                # Check if it's an image
                is_image = file_path.lower().endswith(('.png', '.jpg', '.jpeg'))
                
                if is_image:
                    # Special handling for images using llava format
                    image_prompt = f"\n<image>{file_path}</image>"
                    if current_prompt:
                        # Combine existing prompt with image
                        self.prompt_input.setText(f"{current_prompt}{image_prompt}")
                    else:
                        # Just add image if no prompt exists
                        self.prompt_input.setText(f"Describe this image:{image_prompt}")
                else:
                    # Handle text files
                    with open(file_path, 'r') as file:
                        file_content = file.read()
                        file_ext = Path(file_path).suffix[1:]
                        
                        # Format file content as code block
                        formatted_content = f"\n```{file_ext}\n{file_content}\n```"
                        
                        if current_prompt:
                            # Combine existing prompt with file content
                            self.prompt_input.setText(f"{current_prompt}\nHere's the file content:{formatted_content}")
                        else:
                            # Just add file content if no prompt exists
                            self.prompt_input.setText(f"Analyze this code:{formatted_content}")
                
                self.prompt_input.setFocus()
                
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
        """Show the typing indicator and stop button, start animation"""
        self.is_typing = True
        self.typing_indicator.setVisible(True)
        self.stop_button.setVisible(True)  # Show stop button
        self.typing_dots = 0
        self.typing_timer.start(150)

    def hide_typing_indicator(self):
        """Hide the typing indicator and stop button, stop animation"""
        self.is_typing = False
        self.typing_indicator.setVisible(False)
        self.stop_button.setVisible(False)  # Hide stop button
        self.typing_timer.stop()

    def stop_generation(self):
        """Stop the current generation"""
        if hasattr(self, 'ollama_thread') and self.ollama_thread.isRunning():
            self.ollama_thread.terminate()  # Force stop the thread
            self.ollama_thread.wait()  # Wait for thread to finish
            self.clean_thread(self.ollama_thread)
            self.hide_typing_indicator()
            self.add_system_message("Generation stopped by user")
        
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
        """Refresh model list while preserving current selection"""
        current_model = self.model_combo.currentText()  # Store current selection
        
        self.model_combo.clear()
        self.model_combo.addItem("Loading models...")
        self.model_combo.setEnabled(False)
        self.send_button.setEnabled(False)
        
        # Clear the chat display
        self.messages = []
        self.message_counter = 0
        self.refresh_chat_display()
        
        def on_models_loaded(models):
            self.model_combo.clear()
            if models:
                self.model_combo.addItems(models)
                # Try to restore previous selection
                if current_model in models:
                    index = self.model_combo.findText(current_model)
                    self.model_combo.setCurrentIndex(index)
                self.model_combo.setEnabled(True)
                self.send_button.setEnabled(True)
                selected_model = self.model_combo.currentText()
                self.add_system_message(f"Models refreshed. Using <b>{selected_model}</b>")
            else:
                self.model_combo.addItem("No models found")
                self.add_system_message("No models found. Please pull a model using 'ollama pull <model>' command.")
        
        self.model_thread = ModelListThread()
        self.model_thread.models_signal.connect(on_models_loaded)
        self.model_thread.error_signal.connect(self.show_model_error)
        self.model_thread.finished.connect(lambda: self.clean_thread(self.model_thread))
        self.active_threads.append(self.model_thread)
        self.model_thread.start()
    
    def update_model_list(self, models):
        self.model_combo.clear()
        if models:
            self.model_combo.addItems(models)
            self.model_combo.setCurrentIndex(0)
            selected_model = self.model_combo.currentText()
            self.add_system_message(f"Ready to chat using <b>{selected_model}</b>. Type /help for available commands.")
        else:
            self.model_combo.addItem("No models found")
            self.add_system_message("No models found. Please pull a model using 'ollama pull <model>' command.")
        
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
        chat_html = """
        <style>
            .message-container {
                padding: 15px;
                margin: 10px 0;
                border-radius: 8px;
            }
            
            .user-message {
                background-color: rgba(68, 71, 90, 0.3);
                border-left: 3px solid #BD93F9;
            }
            
            .assistant-message {
                background-color: rgba(40, 42, 54, 0.3);
                border-left: 3px solid #50FA7B;
            }
            
            .system-message {
                background-color: rgba(68, 71, 90, 0.2);
                color: #6272A4;
            }
            
            .message-header {
                font-weight: bold;
                margin-bottom: 10px;
                color: #F8F8F2;
                display: flex;
                justify-content: space-between;
            }
            
            .message-content {
                color: #F8F8F2;
                line-height: 1.5;
            }
            
            .code-block {
                margin: 10px 0;
                border-radius: 8px;
                overflow: hidden;
                border: 1px solid #444;
            }
            
            .code-header {
                background-color: #343746;
                padding: 8px 15px;
                color: #f8f8f2;
                display: flex;
                justify-content: space-between;
                align-items: center;
            }
            
            pre {
                background-color: #282A36;
                color: #f8f8f2;
                padding: 15px;
                margin: 0;
                overflow-x: auto;
                font-family: 'Consolas', 'Monaco', monospace;
            }
            
            code {
                font-family: 'Consolas', 'Monaco', monospace;
                color: #f8f8f2;
            }
            
            .copy-button {
                color: #6272A4;
                cursor: pointer;
                padding: 4px 8px;
                border-radius: 4px;
                background: rgba(255,255,255,0.1);
                border: none;
                font-size: 0.9em;
            }
        </style>
        """

        for message in self.messages:
            message_id = f"msg-{message['id']}"
            
            if message["type"] == "system":
                chat_html += f"""
                <div class="message-container system-message" id="{message_id}">
                    <div class="message-header">System</div>
                    <div class="message-content">{message["content"]}</div>
                </div>
                """
            
            elif message["type"] == "user":
                chat_html += f"""
                <div class="message-container user-message" id="{message_id}">
                    <div class="message-header">You</div>
                    <div class="message-content">{format_code_blocks(message["content"])}</div>
                </div>
                """
            
            elif message["type"] == "assistant":
                chat_html += f"""
                <div class="message-container assistant-message" id="{message_id}">
                    <div class="message-header">Ollama ({message["model"]})</div>
                    <div class="message-content">{format_code_blocks(message["content"])}</div>
                </div>
                """

        self.chat_display.setHtml(chat_html)
        
        # Scroll to bottom
        scrollbar = self.chat_display.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())
    
    def send_prompt(self):
        prompt = self.prompt_input.text().strip()
        if not prompt:
            return
            
        # Handle commands first
        if prompt.startswith('/'):
            self.prompt_input.clear()
            self.prompt_input.setFocus()
            
            if prompt == '/help':
                help_html = """
                <div style="padding: 15px; margin: 10px 0; border-radius: 8px; background-color: rgba(150, 150, 150, 0.1);">
                    <h3>Available Commands:</h3>
                    <ul>
                        <li><code>/help</code> - Show this help message</li>
                        <li><code>/models</code> - Refresh available models</li>
                        <li><code>/clear</code> - Clear chat history</li>
                        <li><code>/save [name]</code> - Save conversation</li>
                        <li><code>/load [name]</code> - Load saved conversation</li>
                        <li><code>/list</code> - List saved conversations</li>
                        <li><code>/swap [model]</code> - Switch to different model</li>
                    </ul>
                </div>
                """
                self.add_system_message(help_html)
                return
                
            elif prompt == '/models':
                self.fetch_models()
                return
                
            elif prompt == '/clear':
                self.messages = []
                self.message_counter = 0
                self.refresh_chat_display()
                return
                
            elif prompt.startswith('/save'):
                name = prompt.replace('/save', '').strip()
                self.save_conversation(name if name else None)
                return
                
            elif prompt.startswith('/load'):
                name = prompt.replace('/load', '').strip()
                if name:
                    self.load_conversation(name)
                else:
                    self.add_system_message("Please specify a conversation name to load")
                return
                
            elif prompt == '/list':
                self.list_conversations()
                return
                
            elif prompt.startswith('/swap'):
                model_name = prompt.replace('/swap', '').strip()
                if model_name:
                    index = self.model_combo.findText(model_name)
                    if index >= 0:
                        self.model_combo.setCurrentIndex(index)
                        self.add_system_message(f"Switched to model: {model_name}")
                    else:
                        self.add_system_message(f"Model '{model_name}' not found")
                return

            elif prompt.startswith('/embed'):
                model = prompt.replace('/embed', '').strip()
                if not model:
                    model = 'nomic-embed-text'  # default model
                self.setup_embedding(model)
                return

            elif prompt.startswith('/search'):
                query = prompt.replace('/search', '').strip()
                if query:
                    self.search_conversations(query)
                else:
                    self.add_system_message("Please provide a search query")
                return

            else:
                self.add_system_message(f"Unknown command: {prompt}")
                return

        # Handle regular message to Ollama
        selected_model = self.model_combo.currentText()
        if selected_model in ["Loading models...", "No models found", "Error loading models"]:
            self.add_system_message("Please select a valid model first")
            return

        # Clear input and add user message
        self.prompt_input.clear()
        self.add_user_message(prompt)
        
        # Create assistant message and get its ID
        message_id = self.add_assistant_message(selected_model)
        
        # Show typing indicator
        self.show_typing_indicator()
        
        # Stop any existing generation
        if hasattr(self, 'ollama_thread') and self.ollama_thread.isRunning():
            self.ollama_thread.terminate()
            self.ollama_thread.wait()
            self.clean_thread(self.ollama_thread)

        # Create and start Ollama thread
        self.ollama_thread = OllamaThread(prompt, selected_model)
        self.ollama_thread.response_signal.connect(
            lambda content: self.update_assistant_message(message_id, content)
        )
        self.ollama_thread.error_signal.connect(
            lambda error: self.add_system_message(f"Error: {error}")
        )
        self.ollama_thread.complete_signal.connect(self.hide_typing_indicator)
        self.ollama_thread.finished.connect(
            lambda: self.clean_thread(self.ollama_thread)
        )
        
        self.active_threads.append(self.ollama_thread)
        self.ollama_thread.start()

    def save_conversation(self, name=None):
        """Save current conversation to parquet file"""
        if not name:
            name = f"conversation_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Convert messages to DataFrame
        df = pd.DataFrame(self.messages)
        
        # Add embeddings if enabled
        if hasattr(self, 'embedding_model'):
            df['embedding'] = df['content'].apply(
                lambda x: ollama.embed(model=self.embedding_model, input=x)['embedding']
            )
        
        # Save to parquet
        save_path = os.path.join(SAVE_DIR, f"{name}.parquet")
        df.to_parquet(save_path)
        self.add_system_message(f"Conversation saved as: {name}")

    def load_conversation(self, name):
        """Load conversation from parquet file"""
        try:
            load_path = os.path.join(SAVE_DIR, f"{name}.parquet")
            if not os.path.exists(load_path):
                self.add_system_message(f"Conversation file not found: {name}")
                return
                
            df = pd.read_parquet(load_path)
            
            # Convert DataFrame back to message format
            messages = []
            for _, row in df.iterrows():
                message = {
                    "id": f"{row['type']}-{len(messages)}",
                    "type": row['type'],
                    "content": row['content']
                }
                if row['type'] == 'assistant':
                    message['model'] = row.get('model', 'unknown')
                messages.append(message)
                
            self.messages = messages
            self.message_counter = len(messages)
            self.refresh_chat_display()
            self.add_system_message(f"Loaded conversation: {name}")
            
        except Exception as e:
            self.add_system_message(f"Error loading conversation: {str(e)}")

    def list_conversations(self):
        """List all saved conversations"""
        files = [f.stem for f in Path(SAVE_DIR).glob("*.parquet")]
        if files:
            msg = "<b>Saved conversations:</b><ul>"
            for f in files:
                msg += f"<li>{f}</li>"
            msg += "</ul>"
        else:
            msg = "No saved conversations found"
        self.add_system_message(msg)

    def setup_embedding(self, model='nomic-embed-text'):
        """Setup embedding model"""
        self.embedding_model = model
        # Create embeddings collection if needed
        if not hasattr(self, 'embeddings_db'):
            self.embeddings_db = []
        self.add_system_message(f"Embedding model set to: {model}")

    def create_embedding(self, text):
        """Create embedding for text using current model"""
        if hasattr(self, 'embedding_model'):
            try:
                embedding = ollama.embed(model=self.embedding_model, input=text)
                return embedding['embedding']
            except Exception as e:
                print(f"Error creating embedding: {e}")
        return None

    def search_conversations(self, query, k=3):
        """Search through saved conversations using embeddings"""
        if not hasattr(self, 'embedding_model'):
            self.add_system_message("Please set up embedding model first using /embed command")
            return
            
        query_embedding = ollama.embed(model=self.embedding_model, input=query)['embedding']
        
        results = []
        for f in Path(SAVE_DIR).glob("*.parquet"):
            df = pd.read_parquet(f)
            if 'embedding' in df.columns:
                for _, row in df.iterrows():
                    if 'embedding' in row:
                        similarity = np.dot(query_embedding, row['embedding'])
                        results.append((similarity, row['content'], f.stem))
        
        results.sort(reverse=True)
        
        msg = f"<b>Top {k} results for: '{query}'</b><br><br>"
        for sim, content, conv in results[:k]:
            msg += f"<b>From {conv}</b> (similarity: {sim:.2f})<br>{content}<br><br>"
        
        self.add_system_message(msg)

    def on_model_changed(self, index):
        """Handle model selection changes"""
        if index >= 0:  # Ensure valid index
            selected_model = self.model_combo.itemText(index)
            if selected_model not in ["Loading models...", "No models found", "Error loading models"]:
                self.messages = []  # Clear chat history
                self.message_counter = 0
                self.refresh_chat_display()
                self.add_system_message(f"Switched to model: <b>{selected_model}</b>")

    def store_conversation_chunk(self, messages, chunk_size=3):
        """Store conversation chunk to parquet with embeddings"""
        df = pd.DataFrame(messages[-chunk_size:])  # Only store last N messages
        
        if hasattr(self, 'embedding_model'):
            df['embedding'] = df['content'].apply(
                lambda x: ollama.embed(model=self.embedding_model, input=x)['embedding']
            )
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        chunk_path = os.path.join(SAVE_DIR, f"chunk_{timestamp}.parquet")
        df.to_parquet(chunk_path)
        return chunk_path

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = OllamaChatApp()
    sys.exit(app.exec())