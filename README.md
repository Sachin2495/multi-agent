# 🦙 Multi-Agent Llama3 Streaming System

A powerful, locally running multi-agent AI system using [Ollama](https://ollama.com/) and Llama3, with real-time streaming output and structured results. Supports text, PDF, and image (OCR) input. Each agent (Research, General, Explainer, Solution) provides a specialized response, all orchestrated by a controller for concurrent, efficient processing.

---

## ✨ Features

- **Runs Locally**: Uses Ollama's Llama3 model on your machine (CPU or GPU).
- **Multi-Agent**: Four specialized agents (Research, General, Explainer, Solution) process your query in parallel.
- **Streaming Output**: See responses token-by-token in real time.
- **Structured Results**: Outputs a structured JSON with all agent responses.
- **Flexible Input**: Accepts direct text, PDF files, or images (with OCR).
- **Knowledge Base**: Optionally enhances explanations with a simple or vector-based document store (ChromaDB, if available).
- **Concurrent Execution**: All agents run in parallel for speed.

---

## 🖥️ Requirements

- **Python 3.9+**
- **Ollama** (with Llama3 model pulled)
- **Windows, Linux, or MacOS**

### Python Dependencies

- `langchain`
- `langchain_community`
- `langchain_chroma` (optional, for vector DB)
- `easyocr`
- `PyPDF2`
- `pillow`
- `requests`

---

## 🚀 Installation

1. **Install Ollama**  
   [Get Ollama](https://ollama.com/download) and follow their instructions for your OS.

2. **Pull the Llama3 Model**  
   ```sh
   ollama pull llama3
   ```

3. **Clone this Repository**  
   ```sh
   git clone https://github.com/Sachin2495/llama3-multiagent.git
   cd llama3-multiagent
   ```

4. **Install Python Dependencies**  
   It's recommended to use a virtual environment:
   ```sh
   python -m venv venv
   venv\Scripts\activate   # On Windows
   # or
   source venv/bin/activate  # On Linux/Mac

   pip install -r requirements.txt
   ```

   If you don't have a `requirements.txt`, use:
   ```sh
   pip install langchain langchain_community easyocr PyPDF2 pillow requests
   # Optional for vector DB:
   pip install langchain_chroma
   ```

5. **Start Ollama**  
   Make sure Ollama is running:
   ```sh
   ollama serve
   ```

---

## 🏃 Usage

Run the main script with your desired input:

- **Text Input**
  ```sh
  python main.py --text "Explain neural networks"
  ```

- **PDF Input**
  ```sh
  python main.py --pdf "document.pdf"
  ```

- **Image Input (OCR)**
  ```sh
  python main.py --img "image.jpg"
  ```

- **Save Output to JSON**
  ```sh
  python main.py --text "What is quantum computing?" --output "results.json"
  ```

---

## 🧠 Agents

- **Research**: Deep research and analysis on your topic.
- **General**: Friendly, conversational assistant.
- **Explainer**: Detailed, educational explanations (optionally enhanced by a knowledge base).
- **Solution**: Complete, well-commented Python code solutions.

---

## ⚡ Example Output

```json
{
  "timestamp": "123456.789",
  "agents": {
    "General": "...",
    "Explainer": "...",
    "Research": "...",
    "Solution": "..."
  },
  "summary": {
    "total_agents": 4,
    "successful_agents": 4,
    "failed_agents": 0
  }
}
```

---

## 🛠️ Troubleshooting

- Make sure Ollama is running and the Llama3 model is pulled.
- For GPU acceleration, Ollama will use your GPU automatically if available and supported.
- If vector DB components are missing, the system will fall back to a simple keyword-based document store.

---

## 📄 License

MIT License

---

## 🙏 Credits

- [Ollama](https://ollama.com/)
- [LangChain](https://python.langchain.com/)
- [EasyOCR](https://github.com/JaidedAI/EasyOCR)
