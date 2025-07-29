# AI Translation Experiment

This project was a 4-week sprint of rapid iteration, prototyping, and regular stakeholder presentations focused on domain-specific machine translation using transformers/LLMs. I've learned that LLM pipeline design is deeply influenced by large differences in infrastructure constraints (costs, setup time, maintenance). Light weight RAG methods with prompt injection can outperform state of the art tools (google-translate) when domain tuning matters.

🔒 Note: Due to proprietary restrictions, exact details of subject matter cannot be shared. This repo provides redacted/cleaned notebooks with visuals on technical details, and model performance.

🚀 Methods Explored
1. 🏗️ Retrieval-Augmented Generation (RAG + Closed-Source Models)

 - Text embeddings: (BERT + LangChain)
 - Zero-shot, and multi-shot prompting using proprietary APIs
 - Multi-shot prompt injection with retrieval from a vector database

2. 🧠 Fine-Tuning (Open-Source Models)

 - Model: Helsinki-NLP Opus MT
 - Fine-tuning on parallel corpora with PEFT and LoRA techniques
 - Grid search over hyperparameters with structured evaluation

⚙️ Infrastructure & Tooling
 - Fine-tuning experiments were run on AWS EC2 (g6.xlarge) with NVIDIA T4 GPU
 - Vector database and RAG components were hosted in-memory for rapid evaluation cycles

