# GenAI Learning Repository

A comprehensive learning repository for Generative AI and LangChain, covering everything from basics to advanced implementations.

## 📋 Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Folder Descriptions](#folder-descriptions)
- [Key Technologies](#key-technologies)
- [Quick Start](#quick-start)

## Overview

This repository is a structured learning path for understanding Generative AI concepts and implementing them using LangChain. It covers everything from fundamental concepts to advanced chains, runnables, and real-world applications.

## Project Structure

```
GenAIx/
├── 1Basics/                  # Foundational concepts
├── 2Components/              # Core LangChain components
├── 3Models/                  # Language and Chat models with embeddings
├── 4Prompts/                 # Prompt engineering and templates
├── 5Output/                  # Output structures and schemas
├── 6Output_Parser/           # Output parsing strategies
├── 7Chains/                  # Sequential and conditional chains
├── 8Runnable1/               # Runnable interfaces (notebooks)
├── 9Runnable2/               # Advanced Runnable patterns
├── 10DocumentLoader/         # Document loading utilities
├── 11TextSplitters/          # Text splitting strategies
├── requirements.txt          # Project dependencies
└── README.md                 # This file
```

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup Instructions

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd GenAIx
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv genai_venv
   ```

3. **Activate the virtual environment:**
   - **Windows (PowerShell):**
     ```powershell
     .genai_venv\Scripts\Activate.ps1
     ```
   - **Windows (Command Prompt):**
     ```cmd
     genai_venv\Scripts\activate.bat
     ```
   - **macOS/Linux:**
     ```bash
     source genai_venv/bin/activate
     ```

4. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Folder Descriptions

### 📚 1Basics
Introduces fundamental LangChain concepts:
- What is LangChain and why it's needed
- Semantic search fundamentals
- **Files:** `notes.txt`

### 🔧 2Components
Core LangChain components and utilities:
- **Files:** `notes.txt`

### 🤖 3Models
Language and Chat model implementations:
- **LLM Models:** General-purpose language models
- **Chat Models:** Conversational AI models
- **Implementations:** OpenAI, Anthropic, Google Gemini, Hugging Face, Local models
- **Embeddings:** Document and query embeddings (OpenAI, Hugging Face)
- **Files:**
  - `llm_demo.py` - LLM demonstrations
  - `chat_model_openai.py`, `chat_model_anthropic.py`, `chat_model_gemini.py` - Provider-specific implementations
  - `hugging_chat_model.py` - Hugging Face integration
  - `chat_local.py` - Local model implementations
  - `embedding/` - Embedding utilities and document similarity

### 💬 4Prompts
Prompt engineering and template design:
- Text-based and multimodal prompts
- Chat prompt templates
- **Files:**
  - `chat_prompt_template.py` - Template implementation
  - `prompt_template_generator.py` - Dynamic template generation
  - `chatbot.py` - Chatbot implementation
  - `message.py`, `message_placeholder.py` - Message handling
  - `research_paper_analysis.py` - Real-world application
  - `prompt_ui.py` - User interface for prompts

### 📤 5Output
Output structure and schema definition:
- JSON schemas
- Pydantic models for structured output
- **Files:**
  - `output.py` - Output handling
  - `pydantic_demo.py` - Pydantic demonstrations
  - `pydantic_structure.py` - Structured Pydantic models
  - `structured_typedict.py`, `typedict.py` - Type dictionary implementations

### 🔄 6Output_Parser
Various output parsing strategies:
- JSON output parsing
- Pydantic output parsing
- String output parsing
- **Files:**
  - `jsonOutputParser.py` - JSON parsing
  - `pydanticParser.py` - Pydantic parsing
  - `strOutputParser.py`, `strOutputParser1.py` - String parsing
  - `structured_output.py` - Structured output handling

### ⛓️ 7Chains
Chain composition patterns:
- Simple chains
- Sequential chains
- Conditional chains
- Parallel chains
- **Files:**
  - `simpleChain.py` - Basic chain implementation
  - `sequentialchain.py` - Sequential execution
  - `conditionalChain.py` - Conditional logic
  - `parallelChain.py` - Parallel execution

### 🔌 8Runnable1
Runnable interface fundamentals (Jupyter Notebooks):
- `a.ipynb`, `b.ipynb` - Interactive runnable demonstrations

### 🚀 9Runnable2
Advanced Runnable patterns:
- Runnable branches
- Runnable lambda functions
- Runnable parallel execution
- Runnable passthrough
- Runnable sequences
- **Files:**
  - `runnable_branch.py` - Branch logic
  - `runnable_lambda.py` - Lambda functions
  - `runnable_parallel.py` - Parallel runnables
  - `runnable_passthrough.py` - Passthrough patterns
  - `runnable_sequence.py` - Sequential runnables

### 📄 10DocumentLoader
Document loading utilities:
- CSV loader
- PDF loader
- Text loader
- Web-based loader
- Directory loader
- **Files:**
  - `csv_loader.py` - CSV file loading
  - `pdf_loader.py` - PDF file loading
  - `text_loader.py` - Text file loading
  - `webbase_loader.py` - Web content loading
  - `directory_loader.py` - Directory scanning and loading
  - **Sample Data:** `Social_Network_Ads.csv`, `cricket.txt`

### ✂️ 11TextSplitters
Text splitting and chunking strategies:
- (Implementation examples for text segmentation)

## Key Technologies

### Language Models & APIs
- **OpenAI** - GPT models and embeddings
- **Anthropic** - Claude models
- **Google** - Gemini and PaLM models
- **Hugging Face** - Open-source models and transformers
- **Local Models** - On-premise LLM deployment

### Core Frameworks
- **LangChain** - LLM orchestration framework
- **LangChain-Core** - Core abstractions
- **Python** - Primary programming language

### ML & Data Tools
- **NumPy** - Numerical computing
- **Scikit-learn** - Machine learning utilities
- **Transformers** - Hugging Face models
- **Streamlit** - Web application framework

### Dependencies Summary
See `requirements.txt` for complete dependency list.

## Quick Start

### 1. Run a Simple Chat Model
```bash
python 3Models/chat_model_openai.py
```

### 2. Test Embeddings
```bash
python 3Models/embedding/1_embedding_openai_query.py
```

### 3. Create a Chatbot
```bash
python 4Prompts/chatbot.py
```

### 4. Build a Chain
```bash
python 7Chains/sequentialchain.py
```

### 5. Load Documents
```bash
python 10DocumentLoader/csv_loader.py
```

## Environment Setup

Create a `.env` file in the root directory with your API keys:

```
OPENAI_API_KEY=your_key_here
ANTHROPIC_API_KEY=your_key_here
GOOGLE_API_KEY=your_key_here
HUGGINGFACEHUB_API_TOKEN=your_token_here
```

## Learning Path Recommendation

1. Start with **1Basics** - Understand LangChain fundamentals
2. Move to **2Components** - Learn core components
3. Explore **3Models** - Try different LLMs
4. Practice **4Prompts** - Master prompt engineering
5. Study **5Output & 6Output_Parser** - Handle structured outputs
6. Build **7Chains** - Compose complex workflows
7. Master **8Runnable1 & 9Runnable2** - Advanced execution patterns
8. Implement **10DocumentLoader & 11TextSplitters** - Handle real-world data

## Contributing

Feel free to enhance examples, add new implementations, or improve documentation.

## License

This project is for educational purposes.

## Notes

- Always activate the virtual environment before running scripts
- Set up environment variables for API keys
- Refer to individual folder notes.txt for specific topics
- Notebooks in 8Runnable1 are best explored interactively