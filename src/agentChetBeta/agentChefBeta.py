import sys
import os
import re
import time
import json
import pandas as pd
import requests
import tarfile
import gzip
import shutil
from pathlib import Path
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                           QLabel, QLineEdit, QPushButton, QTextEdit, QProgressBar,
                           QComboBox, QTabWidget, QFileDialog, QCheckBox, QSpinBox)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer
import ollama

class ArxivFetcher(QThread):
    progress_signal = pyqtSignal(str)
    complete_signal = pyqtSignal(dict)
    error_signal = pyqtSignal(str)
    
    def __init__(self, arxiv_id, download_path):
        super().__init__()
        self.arxiv_id = arxiv_id
        self.download_path = Path(download_path)
        self.download_path.mkdir(parents=True, exist_ok=True)
    
    def run(self):
        try:
            self.progress_signal.emit(f"Fetching metadata for {self.arxiv_id}...")
            paper_info = self.fetch_paper_info()
            
            self.progress_signal.emit(f"Downloading source files for {self.arxiv_id}...")
            paper_dir = self.download_source()
            
            if paper_dir:
                self.progress_signal.emit("Extracting LaTeX content...")
                latex_content = self.extract_latex(paper_dir)
                if latex_content:
                    paper_info['latex_content'] = latex_content
                    paper_info['source_path'] = str(paper_dir)
                    self.complete_signal.emit(paper_info)
                else:
                    self.error_signal.emit("Failed to extract LaTeX content")
            else:
                self.error_signal.emit("Failed to download source files")
                
        except Exception as e:
            self.error_signal.emit(f"Error: {str(e)}")
    
    def extract_arxiv_id(self, url_or_id):
        """Extract arXiv ID from a URL or direct ID string."""
        patterns = [
            r'arxiv.org/abs/([\w.-]+)',
            r'arxiv.org/pdf/([\w.-]+)',
            r'^([\w.-]+)$'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, url_or_id)
            if match:
                return match.group(1)
        
        return None
    
    def fetch_paper_info(self):
        """Fetch paper metadata from arXiv API."""
        import urllib.request
        import urllib.parse
        import xml.etree.ElementTree as ET
        from datetime import datetime
        
        base_url = 'http://export.arxiv.org/api/query'
        query_params = {
            'id_list': self.arxiv_id,
            'max_results': 1
        }
        
        url = f"{base_url}?{urllib.parse.urlencode(query_params)}"
        
        with urllib.request.urlopen(url) as response:
            xml_data = response.read().decode('utf-8')
        
        root = ET.fromstring(xml_data)
        namespaces = {
            'atom': 'http://www.w3.org/2005/Atom',
            'arxiv': 'http://arxiv.org/schemas/atom'
        }
        
        entry = root.find('atom:entry', namespaces)
        if entry is None:
            raise ValueError("No paper found with the provided ID")
        
        paper_info = {
            'arxiv_id': self.arxiv_id,
            'title': entry.find('atom:title', namespaces).text.strip(),
            'authors': [author.find('atom:name', namespaces).text 
                       for author in entry.findall('atom:author', namespaces)],
            'abstract': entry.find('atom:summary', namespaces).text.strip(),
            'published': entry.find('atom:published', namespaces).text,
            'categories': [cat.get('term') for cat in entry.findall('atom:category', namespaces)],
        }
        
        return paper_info
    
    def download_source(self):
        """Download and extract source files for a paper."""
        import urllib.request
        
        # Construct source URL
        source_url = f"https://arxiv.org/e-print/{self.arxiv_id}"
        paper_dir = self.download_path / self.arxiv_id
        paper_dir.mkdir(exist_ok=True)
        
        try:
            # Download source file
            temp_file = paper_dir / "temp_source"
            with urllib.request.urlopen(source_url) as response:
                with open(temp_file, 'wb') as f:
                    f.write(response.read())

            # Try to extract as tar.gz
            try:
                with tarfile.open(temp_file, 'r:gz') as tar:
                    tar.extractall(path=paper_dir)
                    self.progress_signal.emit("Extracted tar.gz source")
            except tarfile.ReadError:
                # If not tar.gz, try as gzip
                try:
                    with gzip.open(temp_file, 'rb') as gz:
                        with open(paper_dir / 'main.tex', 'wb') as f:
                            f.write(gz.read())
                    self.progress_signal.emit("Extracted gzip source")
                except Exception:
                    self.progress_signal.emit("Source is not in standard format, saving as-is")
                    shutil.copy(temp_file, paper_dir / "source_file")

            # Clean up temp file
            if temp_file.exists():
                temp_file.unlink()
            return paper_dir

        except Exception as e:
            self.progress_signal.emit(f"Error downloading source: {str(e)}")
            return None
    
    def extract_latex(self, paper_dir):
        """Extract LaTeX content from source files."""
        latex_content = []
        
        # Find all .tex files
        tex_files = list(paper_dir.glob('**/*.tex'))
        if not tex_files:
            tex_files = list(paper_dir.glob('**/*'))  # Try all files if no .tex found
            
        # Read and concatenate all .tex files
        for tex_file in tex_files:
            try:
                with open(tex_file, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                    latex_content.append(f"% From file: {tex_file.name}\n{content}\n")
            except Exception as e:
                self.progress_signal.emit(f"Failed to read {tex_file.name}: {str(e)}")
                continue
                
        return '\n'.join(latex_content) if latex_content else None

class DataGenerator(QThread):
    progress_signal = pyqtSignal(str)
    complete_signal = pyqtSignal(pd.DataFrame)
    error_signal = pyqtSignal(str)
    
    def __init__(self, paper_info, chunk_size, model, num_samples):
        super().__init__()
        self.paper_info = paper_info
        self.chunk_size = chunk_size
        self.model = model
        self.num_samples = num_samples
        
    def run(self):
        try:
            self.progress_signal.emit("Starting data generation process...")
            
            # Extract content chunks
            latex_content = self.paper_info.get('latex_content', '')
            chunks = self.chunk_latex(latex_content, self.chunk_size)
            self.progress_signal.emit(f"Extracted {len(chunks)} chunks from LaTeX")
            
            # Generate conversations
            conversations = []
            for i, chunk in enumerate(chunks):
                if i >= self.num_samples:
                    break
                    
                self.progress_signal.emit(f"Generating conversation {i+1}/{min(len(chunks), self.num_samples)}...")
                
                # Generate conversation using Ollama
                conversation = self.generate_conversation(chunk)
                if conversation:
                    conversations.append(conversation)
                    self.progress_signal.emit(f"Generated conversation {i+1}")
                    
            # Create dataframe
            if conversations:
                df = pd.DataFrame({'conversations': conversations})
                self.complete_signal.emit(df)
            else:
                self.error_signal.emit("No conversations were generated")
                
        except Exception as e:
            self.error_signal.emit(f"Error: {str(e)}")
    
    def chunk_latex(self, content, chunk_size):
        """Split LaTeX content into chunks"""
        # Clean content - remove comments, normalize whitespace
        content = re.sub(r'%.*$', '', content, flags=re.MULTILINE)
        content = re.sub(r'\s+', ' ', content)
        
        # Extract meaningful sections (abstract, introduction, methods, etc.)
        sections = re.split(r'\\(section|chapter){([^}]+)}', content)
        
        # Create chunks
        chunks = []
        current_chunk = ""
        
        for section in sections:
            if len(current_chunk) + len(section) > chunk_size:
                chunks.append(current_chunk)
                current_chunk = section
            else:
                current_chunk += section
                
        if current_chunk:
            chunks.append(current_chunk)
            
        return chunks
    
    def generate_conversation(self, chunk):
        """Generate conversational data from content chunk"""
        system_prompt = f"""You are an assistant helping to create synthetic training data. 
        Generate a realistic conversation between a human and an AI assistant about the following research content:
        
        {chunk[:500]}... [content truncated for brevity]
        
        The conversation should:
        1. Include 3-5 turns (human question, AI response).
        2. Be related to the research topic.
        3. Show the human asking questions about the research and the AI providing helpful responses.
        4. Format the output as a JSON list with "from" (either "human" or "gpt") and "value" fields.
        
        Return ONLY the JSON array without explanations or markdown formatting."""
        
        try:
            response = ollama.chat(
                model=self.model,
                messages=[{"role": "system", "content": system_prompt}],
            )
            
            content = response['message']['content']
            
            # Extract JSON from the response
            json_match = re.search(r'\[\s*{\s*"from":.+}\s*\]', content, re.DOTALL)
            if json_match:
                conversation_json = json_match.group(0)
                # Validate and clean JSON
                try:
                    conversation = json.loads(conversation_json)
                    return conversation
                except json.JSONDecodeError:
                    self.progress_signal.emit("Error parsing JSON response, trying to clean...")
                    # Try to clean common JSON format issues
                    cleaned_json = re.sub(r'(\w+):', r'"\1":', conversation_json)
                    cleaned_json = re.sub(r'\'', r'"', cleaned_json)
                    try:
                        conversation = json.loads(cleaned_json)
                        return conversation
                    except:
                        return None
            else:
                self.progress_signal.emit("JSON format not found in response")
                return None
                
        except Exception as e:
            self.progress_signal.emit(f"Error generating conversation: {str(e)}")
            return None

class AgentChefApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.papers = {}  # Store paper information
        self.init_ui()
        self.fetch_ollama_models()
        
    def init_ui(self):
        self.setWindowTitle("Agent Chef - AI Training Data Generator")
        self.setGeometry(100, 100, 900, 700)
        
        # Main layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout(main_widget)
        
        # Create tabs
        tabs = QTabWidget()
        main_layout.addWidget(tabs)
        
        # Paper Download Tab
        download_tab = QWidget()
        tabs.addTab(download_tab, "1. Paper Download")
        
        download_layout = QVBoxLayout(download_tab)
        
        # ArXiv ID input
        id_layout = QHBoxLayout()
        id_layout.addWidget(QLabel("ArXiv URL or ID:"))
        self.arxiv_input = QLineEdit()
        self.arxiv_input.setPlaceholderText("e.g., 2307.09288 or https://arxiv.org/abs/2307.09288")
        id_layout.addWidget(self.arxiv_input)
        self.fetch_button = QPushButton("Fetch Paper")
        self.fetch_button.clicked.connect(self.fetch_paper)
        id_layout.addWidget(self.fetch_button)
        download_layout.addLayout(id_layout)
        
        # Paper info display
        download_layout.addWidget(QLabel("Paper Information:"))
        self.paper_info = QTextEdit()
        self.paper_info.setReadOnly(True)
        download_layout.addWidget(self.paper_info)
        
        # Status and progress
        download_layout.addWidget(QLabel("Status:"))
        self.status_text = QTextEdit()
        self.status_text.setReadOnly(True)
        self.status_text.setMaximumHeight(150)
        download_layout.addWidget(self.status_text)
        
        # Generator Tab
        generator_tab = QWidget()
        tabs.addTab(generator_tab, "2. Data Generation")
        
        generator_layout = QVBoxLayout(generator_tab)
        
        # Model selection
        model_layout = QHBoxLayout()
        model_layout.addWidget(QLabel("Ollama Model:"))
        self.model_combo = QComboBox()
        model_layout.addWidget(self.model_combo)
        self.refresh_button = QPushButton("Refresh")
        self.refresh_button.clicked.connect(self.fetch_ollama_models)
        model_layout.addWidget(self.refresh_button)
        generator_layout.addLayout(model_layout)
        
        # Generation options
        options_layout = QHBoxLayout()
        
        options_layout.addWidget(QLabel("Chunk Size:"))
        self.chunk_size = QSpinBox()
        self.chunk_size.setRange(500, 10000)
        self.chunk_size.setValue(2000)
        options_layout.addWidget(self.chunk_size)
        
        options_layout.addWidget(QLabel("Samples:"))
        self.num_samples = QSpinBox()
        self.num_samples.setRange(1, 100)
        self.num_samples.setValue(5)
        options_layout.addWidget(self.num_samples)
        
        generator_layout.addLayout(options_layout)
        
        # Select paper
        paper_select_layout = QHBoxLayout()
        paper_select_layout.addWidget(QLabel("Select Paper:"))
        self.paper_combo = QComboBox()
        paper_select_layout.addWidget(self.paper_combo)
        generator_layout.addLayout(paper_select_layout)
        
        # Generate button
        self.generate_button = QPushButton("Generate Training Data")
        self.generate_button.clicked.connect(self.generate_data)
        generator_layout.addWidget(self.generate_button)
        
        # Result preview
        generator_layout.addWidget(QLabel("Result Preview:"))
        self.result_preview = QTextEdit()
        self.result_preview.setReadOnly(True)
        generator_layout.addWidget(self.result_preview)
        
        # Generation status
        generator_layout.addWidget(QLabel("Generation Status:"))
        self.generation_status = QTextEdit()
        self.generation_status.setReadOnly(True)
        self.generation_status.setMaximumHeight(150)
        generator_layout.addWidget(self.generation_status)
        
        # Export Tab
        export_tab = QWidget()
        tabs.addTab(export_tab, "3. Export Data")
        
        export_layout = QVBoxLayout(export_tab)
        
        # Data stats
        export_layout.addWidget(QLabel("Generated Dataset Statistics:"))
        self.data_stats = QTextEdit()
        self.data_stats.setReadOnly(True)
        export_layout.addWidget(self.data_stats)
        
        # Export options
        export_options_layout = QHBoxLayout()
        
        self.export_parquet = QCheckBox("Parquet")
        self.export_parquet.setChecked(True)
        export_options_layout.addWidget(self.export_parquet)
        
        self.export_json = QCheckBox("JSON")
        self.export_json.setChecked(True)
        export_options_layout.addWidget(self.export_json)
        
        self.export_csv = QCheckBox("CSV")
        export_options_layout.addWidget(self.export_csv)
        
        export_layout.addLayout(export_options_layout)
        
        # Export button
        self.export_button = QPushButton("Export Dataset")
        self.export_button.clicked.connect(self.export_dataset)
        export_layout.addWidget(self.export_button)
        
        # Export status
        export_layout.addWidget(QLabel("Export Status:"))
        self.export_status = QTextEdit()
        self.export_status.setReadOnly(True)
        self.export_status.setMaximumHeight(150)
        export_layout.addWidget(self.export_status)
    
    def log_status(self, message, area="status"):
        """Log a message to the specified status area"""
        if area == "status":
            self.status_text.append(message)
            self.status_text.verticalScrollBar().setValue(self.status_text.verticalScrollBar().maximum())
        elif area == "generation":
            self.generation_status.append(message)
            self.generation_status.verticalScrollBar().setValue(self.generation_status.verticalScrollBar().maximum())
        elif area == "export":
            self.export_status.append(message)
            self.export_status.verticalScrollBar().setValue(self.export_status.verticalScrollBar().maximum())
    
    def fetch_ollama_models(self):
        """Fetch available Ollama models"""
        self.model_combo.clear()
        self.log_status("Fetching available Ollama models...")
        
        try:
            result = ollama.list()
            models = []
            
            if hasattr(result, 'models'):
                models = [model.model for model in result.models] 
            elif isinstance(result, dict) and 'models' in result:
                models = [model.get('name', '') for model in result['models']]
            
            if models:
                self.model_combo.addItems(models)
                self.log_status(f"Found {len(models)} models: {', '.join(models[:3])}...")
            else:
                self.log_status("No models found. Please install models with 'ollama pull <model>'")
        except Exception as e:
            self.log_status(f"Error fetching models: {str(e)}")
            self.model_combo.addItem("Error loading models")
    
    def fetch_paper(self):
        """Fetch paper from ArXiv"""
        arxiv_url_or_id = self.arxiv_input.text().strip()
        if not arxiv_url_or_id:
            self.log_status("Please enter an ArXiv URL or ID")
            return
        
        # Extract ArXiv ID
        arxiv_id = None
        patterns = [
            r'arxiv.org/abs/([\w.-]+)',
            r'arxiv.org/pdf/([\w.-]+)',
            r'^([\w.-]+)$'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, arxiv_url_or_id)
            if match:
                arxiv_id = match.group(1)
                break
        
        if not arxiv_id:
            self.log_status("Invalid ArXiv URL or ID")
            return
        
        # Clear previous info
        self.paper_info.clear()
        self.status_text.clear()
        self.log_status(f"Fetching paper {arxiv_id}...")
        
        # Create download directory
        download_path = Path("./papers")
        download_path.mkdir(exist_ok=True)
        
        # Start fetcher thread
        self.fetcher = ArxivFetcher(arxiv_id, download_path)
        self.fetcher.progress_signal.connect(lambda msg: self.log_status(msg))
        self.fetcher.complete_signal.connect(self.handle_paper_fetched)
        self.fetcher.error_signal.connect(lambda err: self.log_status(f"Error: {err}"))
        self.fetcher.start()
    
    def handle_paper_fetched(self, paper_info):
        """Handle fetched paper information"""
        self.log_status("Paper fetched successfully!")
        
        # Display paper info
        info_text = f"Title: {paper_info['title']}\n\n"
        info_text += f"Authors: {', '.join(paper_info['authors'])}\n\n"
        info_text += f"Categories: {', '.join(paper_info['categories'])}\n\n"
        info_text += f"Published: {paper_info['published']}\n\n"
        info_text += f"Abstract: {paper_info['abstract']}\n\n"
        
        if 'latex_content' in paper_info:
            info_text += f"LaTeX content length: {len(paper_info['latex_content'])} characters"
        
        self.paper_info.setText(info_text)
        
        # Store paper info
        paper_id = f"{paper_info['arxiv_id']} - {paper_info['title'][:30]}..."
        self.papers[paper_id] = paper_info
        
        # Update paper selector
        current_items = [self.paper_combo.itemText(i) for i in range(self.paper_combo.count())]
        if paper_id not in current_items:
            self.paper_combo.addItem(paper_id)
            self.paper_combo.setCurrentText(paper_id)
    
    def generate_data(self):
        """Generate training data from paper"""
        selected_paper = self.paper_combo.currentText()
        if not selected_paper or selected_paper not in self.papers:
            self.log_status("Please select a paper first", "generation")
            return
        
        selected_model = self.model_combo.currentText()
        if not selected_model or selected_model == "Error loading models":
            self.log_status("Please select a valid Ollama model", "generation")
            return
        
        # Clear previous results
        self.result_preview.clear()
        self.generation_status.clear()
        
        paper_info = self.papers[selected_paper]
        chunk_size = self.chunk_size.value()
        num_samples = self.num_samples.value()
        
        self.log_status(f"Starting data generation with {selected_model}...", "generation")
        self.log_status(f"Generating {num_samples} samples with chunk size {chunk_size}", "generation")
        
        # Start generator thread
        self.generator = DataGenerator(paper_info, chunk_size, selected_model, num_samples)
        self.generator.progress_signal.connect(lambda msg: self.log_status(msg, "generation"))
        self.generator.complete_signal.connect(self.handle_data_generated)
        self.generator.error_signal.connect(lambda err: self.log_status(f"Error: {err}", "generation"))
        self.generator.start()
    
    def handle_data_generated(self, df):
        """Handle generated training data"""
        self.log_status("Data generation completed!", "generation")
        
        # Store the dataframe
        self.generated_df = df
        
        # Preview results
        if len(df) > 0:
            preview = json.dumps(df['conversations'][0], indent=2)
            self.result_preview.setText(preview)
            
            # Update stats
            stats_text = f"Generated {len(df)} conversation samples\n\n"
            
            # Calculate average turns per conversation
            total_turns = sum(len(conv) for conv in df['conversations'])
            avg_turns = total_turns / len(df)
            stats_text += f"Average turns per conversation: {avg_turns:.2f}\n\n"
            
            # Count total tokens (rough estimate)
            total_words = sum(sum(len(turn['value'].split()) for turn in conv) for conv in df['conversations'])
            stats_text += f"Total words: {total_words}\n\n"
            
            self.data_stats.setText(stats_text)
        else:
            self.result_preview.setText("No data generated")
    
    def export_dataset(self):
        """Export the generated dataset"""
        if not hasattr(self, 'generated_df') or self.generated_df.empty:
            self.log_status("No data available to export", "export")
            return
        
        # Get export directory
        export_dir = QFileDialog.getExistingDirectory(self, "Select Export Directory")
        if not export_dir:
            return
        
        export_dir = Path(export_dir)
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        base_filename = f"agentchef_dataset_{timestamp}"
        
        self.log_status(f"Exporting dataset to {export_dir}", "export")
        
        try:
            # Export to parquet
            if self.export_parquet.isChecked():
                parquet_path = export_dir / f"{base_filename}.parquet"
                self.generated_df.to_parquet(parquet_path)
                self.log_status(f"Exported to parquet: {parquet_path}", "export")
            
            # Export to JSON
            if self.export_json.isChecked():
                json_path = export_dir / f"{base_filename}.json"
                self.generated_df.to_json(json_path, orient='records')
                self.log_status(f"Exported to JSON: {json_path}", "export")
            
            # Export to CSV
            if self.export_csv.isChecked():
                # For CSV, we need to convert the nested conversations
                flat_data = []
                for i, row in self.generated_df.iterrows():
                    conversation_json = json.dumps(row['conversations'])
                    flat_data.append({'conversation_id': i, 'conversation_json': conversation_json})
                
                flat_df = pd.DataFrame(flat_data)
                csv_path = export_dir / f"{base_filename}.csv"
                flat_df.to_csv(csv_path, index=False)
                self.log_status(f"Exported to CSV: {csv_path}", "export")
            
            self.log_status("Export completed successfully!", "export")
            
        except Exception as e:
            self.log_status(f"Error during export: {str(e)}", "export")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = AgentChefApp()
    window.show()
    sys.exit(app.exec())
