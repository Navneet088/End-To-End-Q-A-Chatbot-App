# 💬 Simple Q&A Chatbot with Groq LLM

> **Ask anything. Get instant answers — powered by Llama via Groq & tracked with LangSmith**

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-LCEL-1C3C3C?style=for-the-badge)
![Groq](https://img.shields.io/badge/Groq-Llama_3.1-FF6B35?style=for-the-badge)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![LangSmith](https://img.shields.io/badge/LangSmith-Tracking-00B140?style=for-the-badge)

---

## 📌 Overview

**Simple Q&A Chatbot** is a clean, minimal AI app that answers any question in real time using **Groq-hosted Llama models** — with full **LangSmith observability** built in.

Designed as a lightweight yet production-aware starter, it lets you tune model behaviour directly from the sidebar — no code changes needed.

---

## ✨ Features

| Feature | Description |
|--------|-------------|
| 🤖 **Model Selector** | Switch between `llama-3.1-8b-instant` and `llama-3.1-16b-instant` from the sidebar |
| 🌡️ **Temperature Control** | Slider from `0.0` (focused) to `1.0` (creative) — tune response style live |
| 📏 **Max Tokens Slider** | Control response length between 50 and 300 tokens |
| 🔐 **Secure API Input** | Password-masked Groq API key field in the sidebar |
| 🔗 **LCEL Chain** | Clean `prompt \| llm \| output_parser` pipeline using LangChain Expression Language |
| 📋 **Prompt Template** | `ChatPromptTemplate` with a system role + human question slot |
| 📡 **LangSmith Tracking** | Every chain invocation is logged to LangSmith for observability & debugging |
| ✂️ **StrOutputParser** | Strips LangChain message wrappers — returns clean plain text |

---

## 🛠️ Tech Stack

- **Python 3.10+**
- **Streamlit** — UI and sidebar controls
- **LangChain (LCEL)** — chain orchestration with `|` pipe syntax
- **langchain-groq** — Groq-hosted Llama 3.1 models
- **LangSmith** — LangChain tracing and project observability
- **python-dotenv** — `.env` secret management

---

## 🚀 Installation & Setup

### 1. Clone the repository

```bash
git clone https://github.com/your-username/simple-qa-chatbot-groq.git
cd simple-qa-chatbot-groq
```

### 2. Install dependencies

```bash
pip install streamlit langchain langchain-groq langchain-core python-dotenv openai
```

### 3. Add your API keys

Create a `.env` file in the root directory:

```env
GROQ_API_KEY=your_groq_api_key_here
LANGCHAIN_API_KEY=your_langsmith_api_key_here
```

### 4. Run the app

```bash
streamlit run app.py
```

---

## 📖 How to Use

1. Open the app at `http://localhost:8501`
2. Enter your **Groq API key** in the sidebar
3. Select a **model** from the dropdown
4. Adjust **Temperature** and **Max Tokens** to your preference
5. Type any question in the input box
6. Read your instant AI-generated answer ✅

---

## 🏗️ How It Works

```
User Question
      │
      ▼
ChatPromptTemplate
  ├─ system: "You are a helpful assistant..."
  └─ human:  "{question}"
      │
      ▼
ChatGroq (selected model via sidebar)
  ├─ temperature  ← sidebar slider
  └─ max_tokens   ← sidebar slider
      │
      ▼
StrOutputParser  ← strips message wrapper
      │
      ▼
Plain Text Answer → Streamlit UI
      │
      ▼
LangSmith  ← logs full trace automatically
```

---

## 🎛️ Sidebar Controls

| Control | Range | Default | Effect |
|---------|-------|---------|--------|
| 🔑 Groq API Key | — | — | Authenticates the LLM request |
| 🤖 Model | 8B / 16B | `llama-3.1-8b-instant` | Speed vs capacity trade-off |
| 🌡️ Temperature | 0.0 – 1.0 | 0.7 | Lower = focused, Higher = creative |
| 📏 Max Tokens | 50 – 300 | 150 | Controls response length |

---

## 📊 LangSmith Observability

This app is configured to log every chain run to **LangSmith** under the project:

```
Simple Q&A Chatbot With Groq LLM
```

You can view full traces — inputs, outputs, latency, token usage — at:

```
https://smith.langchain.com
```

> To disable tracking, remove or comment out the `LANGCHAIN_*` environment variables in `.env`.

---

## 💬 Example Questions

```
"What is the difference between machine learning and deep learning?"
"Explain quantum computing in simple terms"
"What are the SOLID principles in software engineering?"
"How does the transformer architecture work?"
"Write a Python function to reverse a linked list"
```

---

## 📁 Project Structure

```
simple-qa-chatbot-groq/
│
├── app.py              # Main Streamlit application
├── .env                # API keys (not committed to git)
├── requirements.txt    # Python dependencies
└── README.md
```

---

## 📦 requirements.txt

```
streamlit
langchain
langchain-groq
langchain-core
openai
python-dotenv
```

---

## ⚠️ Important Notes

> - `temperature` and `max_tokens` are passed to `generate_response()` but not yet wired into the `ChatGroq` constructor — connect them for full sidebar control.
> - The `openai` import and `openai.api_key` assignment are currently unused — safe to remove unless OpenAI support is planned.
> - LangSmith tracking requires a valid `LANGCHAIN_API_KEY` — the app still works without it, tracing will just be skipped.

---

> Built with ❤️ using **LangChain · Groq · LangSmith · Streamlit**
