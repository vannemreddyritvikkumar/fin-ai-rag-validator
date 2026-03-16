# fin-ai-rag-validator
It tests a RAG (Retrieval-Augmented Generation) pipeline. Instead of just checking if the AI "sounds good," it uses a second LLM (the "Judge") to verify three things called the RAG Triad

# 🏦 Financial RAG Quality Validator

An automated evaluation suite for testing **Retrieval-Augmented Generation (RAG)** pipelines in banking and FinTech environments. 

### 🎯 The Problem
Traditional testing fails to detect **AI Hallucinations**. In FinTech, an AI giving a customer the wrong deadline for a dispute (e.g., "90 days" instead of "60 days") is a critical compliance failure.

### 🧠 The AI Solution
This project implements the **RAG Triad** using the **DeepEval** framework (LLM-as-a-Judge) to objectively grade AI outputs on:
- **Faithfulness:** Does the answer contradict the source knowledge base?
- **Answer Relevancy:** Does the response actually address the user's intent?
- **Contextual Precision:** Did the retriever pull the correct policy document?

### 🛠 Tech Stack
- **Python** (Core Logic)
- **DeepEval** (LLM-as-a-Judge Framework)
- **Pytest** (Test Runner)
- **OpenAI API** (The Evaluator Model)

### 🚀 Quick Start
1. `pip install deepeval pytest python-dotenv`
2. Add your `OPENAI_API_KEY` to a `.env` file.
3. Run tests: `pytest tests/test_rag_quality.py`
