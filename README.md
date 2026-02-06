# chatINDECIMAL — A Document-Grounded AI Chatbot (RAG)

## 📌 About the Project
**chatINDECIMAL** is an AI chatbot built using **Retrieval-Augmented Generation (RAG)** as part of an academic assignment.

Instead of answering from guesswork or general knowledge, this chatbot **reads and understands internal documents** (policies, FAQs, and process notes) and answers questions **only when the information is clearly available**.  
If something is not mentioned in the documents, the bot clearly says so — avoiding hallucinations.

This repository represents the **final and complete assignment submission**.

---

## 🎯 What I Wanted to Achieve
- Build an AI assistant that answers **only from documents**
- Replace keyword search with **semantic search**
- Make the system **safe and trustworthy** by preventing hallucinations
- Support **normal, human-style questions**
- Present the solution using a **chat-style UI**

---

## 🧠 What is RAG (in simple terms)?
Retrieval-Augmented Generation (RAG) works in two steps:

1. **Retrieve**  
   The system first searches through documents to find the most relevant parts using embeddings and vector search.

2. **Generate**  
   A language model then generates an answer **using only those retrieved document parts**.

This means:
- The chatbot does **not rely on its training memory**
- Every answer is grounded in actual documents

---
## ⚙️ How the System Works

User asks a question  
↓  
Question converted into embeddings  
↓  
FAISS searches relevant document chunks  
↓  
Top relevant chunks selected  
↓  
Prompt built using only those chunks  
↓  
LLM generates an answer  
↓  
Answer OR "Information not found in documents."


---
## 📂 Project Structure

chatINDECIMAL/
├── streamlit_app.py      # Chat-style web interface
├── README.md
├── .env                 # API key (ignored by git)
├── .gitignore
├── src/
│   ├── ingest.py        # Reads & chunks documents
│   ├── rag_pipeline.py  # Retrieval + generation logic
├── data/
│   └── documents/       # Markdown policy & FAQ files


## ⚙️ Main Components Explained

### 📄 Document Ingestion
- Reads `.md` files from the documents folder
- Splits them into smaller, meaningful chunks
- Stores metadata like document source

This helps the system find **only the relevant parts** instead of entire documents.

---

### 🔍 Semantic Retrieval
- Uses `sentence-transformers/all-MiniLM-L6-v2`
- Stores embeddings in a FAISS vector index
- Finds the **most semantically similar content**, not just keyword matches

---

### 🤖 Answer Generation (Grounded)
- Uses an LLM through **OpenRouter**
- Strong rules are enforced:
  - Answers must come **only from retrieved context**
  - No guessing or assumptions
  - Uses “Yes” or “No” only when explicitly supported
  - Otherwise replies with:


---

## 💬 Chat Interface (Streamlit)

A chat-style UI is built using **Streamlit** to make the system easy to use.

### Features:
- Chat-like conversation experience
- User and bot message bubbles
- Loading animation (“Thinking…”) while answering
- Session-based chat memory
- Option to view retrieved document context (for transparency)

Run the chatbot UI using:
bash-
python -m streamlit run streamlit_app.py


🔐 Why “Information not found” is a Feature

If the required information is not clearly written in the documents, the system refuses to answer.
This is intentional and important because:

It avoids hallucinations.
It builds trust.
It matches real-world enterprise AI behavior.

🧪 Example Questions
Questions the Bot Can Answer

How does Indecimal ensure transparency?
Can customers track construction progress online?
How are payments released to contractors?

Questions the Bot Correctly Refuses

What is Indecimal’s phone number?
How much does construction cost per square foot?
What is the exact project completion time?

🛠️ Technologies Used

Python 3
Sentence Transformers
FAISS (Vector Database)
OpenRouter (LLM API)
Streamlit

🚀 How to Run the Project
1️⃣ Install required libraries
pip install sentence-transformers faiss-cpu streamlit requests python-dotenv

2️⃣ Create a .env file
OPENROUTER_API_KEY=your_api_key_here

3️⃣ Start the chatbot
python -m streamlit run streamlit_app.py



👤 Author
Dinesh Mardi
