Simple GenAI RAG Application:

Tech stack:

Technology	Purpose
Python 3.12	Application development
Groq API	LLM response generation
LangChain	RAG orchestration
HuggingFace	Generate embeddings
FAISS	Store and retrieve vectors
PyPDF	Read PDF documents

The objective is to help you understand how RAG works practically, including document loading, chunking, embeddings, vector storage, semantic search and LLM response generation.

1. What are we going to build?
Use case: An IT employee asks questions about support procedures.

For example:
- How do I reset my password?
- What should I do if VPN is not connecting?
- How do I raise a critical incident?
- What is the SLA for a P1 incident?

Our application retrieves relevant information from knowledge documents and sends that information to Groq LLM to generate an answer.

Install required Python libraries:

python -m venv venv

# Windows
venv\Scripts\activate

pip install -r requirements.txt
pip install langchain-text-splitters
pip install python-dotenv
pip install langchain-community
pip install langchain-huggingface sentence-transformers
pip install langchain-groq
Test Case 1:
How do I reset my password?

Expected answer:
You can reset your password through the corporate self‑service portal. If you are unable to reset it there, contact the IT support team for assistance.
Test Case 2:
What is the resolution SLA for P1 incidents?

Expected answer: The resolution SLA for P1 incidents is **4 hours**.
Test Case 2:
Which team handles VPN issues?
Expected answer: Network Support Team.

Note:
The example uses a local HuggingFace embedding model, so embedding generation does not require a separate paid embedding API.
For production, we have to add authentication, access control, document-level permissions, retrieval evaluation and proper prompt-injection defences.
Guardrails we can incorporate. Since this application doesn’t require, I didn’t add.
I have shared my Groq API key. Even it’s not advisable to share it, to test my application I have shared it.
Also, checked in my API key in GitHub. I didn’t add it in git ignore for testing my application. Otherwise, you have to use your key in the file. To avoid that, I checked in my key.

