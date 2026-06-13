**# RAG-Based Information Retrieval System 🤖**



**> A fully local, zero-cost Retrieval-Augmented Generation (RAG)** 

**> chatbot for \[Maharaja Agrasen College](https://mac.du.ac.in) —** 

**> operational on 8GB RAM, CPU-only, zero cloud dependency.**



**---**



**## System Architecture**



**```**

**Web Crawler (BS4 + Playwright + PyMuPDF)**

&#x20;       **↓**

**5-Stage Data Pipeline**

**(Scrape → Preprocess → Chunk → Embed → Store)**

&#x20;       **↓**

**FAISS Vector Database (1,470 chunks, 384-dim)**

&#x20;       **↓**

**3-Pass Guardrail (Static → Domain → RAG)**

&#x20;       **↓**

**Llama 3.2 3B (LM Studio) → FastAPI → React Panel**

**```**



**## Tech Stack**



**| Layer | Technology |**

**|---|---|**

**| Web Scraping | BeautifulSoup4, Playwright, PyMuPDF |**

**| Embeddings | all-MiniLM-L6-v2 (Sentence-Transformers) |**

**| Vector Store | FAISS IndexFlatL2 |**

**| LLM | Llama 3.2 3B Instruct Q4\_K\_M (LM Studio) |**

**| Backend | FastAPI (async, Python) |**

**| Frontend | React + Vite + Tailwind CSS |**

**| Database | MongoDB |**



**## Evaluation Results**



**| Metric | Score |**

**|---|---|**

**| Hit Rate@5 | 78.1% |**

**| MRR@5 | 0.5143 |**

**| Context Precision | 84.4% |**

**| BERTScore F1 | 0.7887 |**

**| Faithfulness | 100% |**

**| Off-topic Rejection | 100% |**

**| Hallucination Count | 0 |**

**| Overall Accuracy | 85% |**

**| P50 Latency | 2.3s (CPU-only) |**



**## Setup**



**1. Clone this repo**

**2. Copy `.env.example` → `.env` and fill values**

**3. `pip install -r backend/requirements.txt`**

**4. Start MongoDB locally**

**5. Start LM Studio with Llama 3.2 3B loaded**

**6. `uvicorn api:app --reload` from `/backend`**

**7. `cd admin-panel \&\& npm install \&\& npm run dev`**



**## Research Context**



**B.Sc. (Hons.) Electronics Dissertation — Maharaja Agrasen College,**

**University of Delhi (May 2026).**

**Presented at 9th National Student Academic Congress 2026.**



**\*\*Supervisor:\*\* Prof. Amit Pundir**

**\*\*Co-Supervisor:\*\* Prof. Geetika Jain Saxena**

## License
This project is licensed under the Apache 2.0 License. 
See [LICENSE](LICENSE) for details.
© 2026 Kartikey Tiwari

