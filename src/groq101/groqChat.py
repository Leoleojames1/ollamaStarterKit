import sys
import os
from PyQt6.QtWidgets import (QApplication, QMainWindow, QTextEdit, 
                            QLineEdit, QPushButton, QVBoxLayout, QWidget,
                            QSplitter, QLabel, QHBoxLayout)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QFont, QColor, QPalette
import groq

# Thread for handling Groq API calls without freezing UI
class GroqThread(QThread):
    response_signal = pyqtSignal(str)
    
    def __init__(self, prompt, parent=None):
        super().__init__(parent)
        self.prompt = prompt
        self.api_key = os.environ.get("GROQ_API_KEY", "")
        
    def run(self):
        try:
            client = groq.Groq(api_key=self.api_key)
            completion = client.chat.completions.create(
                model="qwen-2.5-coder-32b",
                messages=[{"role": "user", "content": self.prompt}],
                temperature=0.6,
                max_completion_tokens=4096,
                top_p=0.95,
                stream=True,
                stop=None,
            )
            
            response = ""
            for chunk in completion:
                content = chunk.choices[0].delta.content or ""
                response += content
                self.response_signal.emit(content)
                
        except Exception as e:
            self.response_signal.emit(f"\nError: {str(e)}")


class GroqChatApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.init_ui()
        
    def init_ui(self):
        # Set dark theme colors
        self.setStyleSheet("""
            QMainWindow, QWidget {
                background-color: #1e1e2e;
                color: #cdd6f4;
            }
            QTextEdit, QLineEdit {
                background-color: #313244;
                color: #cdd6f4;
                border: 1px solid #45475a;
                border-radius: 5px;
                padding: 5px;
            }
            QPushButton {
                background-color: #89b4fa;
                color: #1e1e2e;
                border: none;
                border-radius: 5px;
                padding: 8px 15px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #b4befe;
            }
            QPushButton:pressed {
                background-color: #74c7ec;
            }
            QLabel {
                color: #f5c2e7;
                font-weight: bold;
            }
        """)
        
        # Main widget and layout
        main_widget = QWidget()
        main_layout = QVBoxLayout(main_widget)
        
        # Header
        header = QLabel("Groq Chat Interface")
        header.setFont(QFont("Arial", 16))
        header.setAlignment(Qt.AlignmentFlag.AlignCenter)
        main_layout.addWidget(header)
        
        # Chat display
        self.chat_display = QTextEdit()
        self.chat_display.setReadOnly(True)
        self.chat_display.setFont(QFont("Consolas", 10))
        
        # Input area
        input_layout = QHBoxLayout()
        self.prompt_input = QLineEdit()
        self.prompt_input.setPlaceholderText("Enter your prompt here...")
        self.prompt_input.returnPressed.connect(self.send_prompt)
        
        self.send_button = QPushButton("Send")
        self.send_button.clicked.connect(self.send_prompt)
        
        input_layout.addWidget(self.prompt_input, 5)
        input_layout.addWidget(self.send_button, 1)
        
        # Add widgets to main layout
        main_layout.addWidget(self.chat_display, 4)
        main_layout.addLayout(input_layout, 1)
        
        # Set main widget
        self.setCentralWidget(main_widget)
        
        # Window settings
        self.setWindowTitle("Groq Chat")
        self.setGeometry(300, 300, 600, 500)
        self.show()
        
        # Welcome message
        self.chat_display.append("Welcome to Groq Chat! Enter a prompt to begin.")
        self.chat_display.append("Using model: qwen-2.5-coder-32b\n")
        
        # Check for API key
        if not os.environ.get("GROQ_API_KEY"):
            self.chat_display.append("⚠️ GROQ_API_KEY environment variable not found!\n"
                                   "Please set your API key before using this app.")
        
    def send_prompt(self):
        prompt = self.prompt_input.text().strip()
        if not prompt:
            return
            
        # Clear input field
        self.prompt_input.clear()
        
        # Display user message
        self.chat_display.append(f"You: {prompt}\n")
        self.chat_display.append("Groq: ")
        
        # Start thread for API call
        self.groq_thread = GroqThread(prompt)
        self.groq_thread.response_signal.connect(self.update_response)
        self.groq_thread.start()
        
    def update_response(self, content):
        # Update the chat display with streamed content
        self.chat_display.insertPlainText(content)
        # Auto scroll to bottom
        self.chat_display.verticalScrollBar().setValue(
            self.chat_display.verticalScrollBar().maximum()
        )


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = GroqChatApp()
    sys.exit(app.exec())
