# getGPT

Streamlit (and Chainlit) apps for **GET set** retrieval and related analysis around a disease:

- **G** — genetic variants  
- **E** — differentially expressed genes  
- **T** — known drug targets  

Live app: [getgpt.streamlit.app](https://getgpt.streamlit.app)

## Features (Streamlit `getGPT.py`)

- Expert Q&A chat with selectable personas (Biologist, Informatician, Computer Scientist, General Expert)
- Gene analysis combining Open Targets Platform, Open Targets Genetics, and PubMed abstracts (with LLM extraction), plus CSV download of the gene table
- PDF upload with FAISS retrieval QA and summary
- Open Targets Platform / Genetics API connectivity test tab

There is also a `chainlit_app.py` entry point and `chainlit.md` for the Chainlit variant.

## Requirements

See `requirements.txt` (requests, chainlit, openai, langchain, langchain_community, langchain_openai, biopython, xmltodict, pypdf, faiss-cpu). Streamlit is used by `getGPT.py` and should be installed if you run that entry point.

## Setup

```bash
pip install -r requirements.txt
pip install streamlit pandas  # needed for getGPT.py if not already present
```

Configure Streamlit secrets with `OPENAI_API_KEY`. Set a real `Entrez.email` in code before heavy PubMed use.

## Usage

```bash
streamlit run getGPT.py
```

Or run the Chainlit app via `chainlit_app.py` (see Chainlit docs for `chainlit run`).
