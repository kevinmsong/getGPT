# getGPT

Streamlit and Chainlit apps that help researchers assemble a disease's **GET set** from public databases and literature:

- **G**: genetic variants
- **E**: differentially expressed genes
- **T**: known drug targets

Live app: [getgpt.streamlit.app](https://getgpt.streamlit.app)

## Features (Streamlit app, `getGPT.py`)

- Expert Q&A chat with selectable personas (Biologist, Informatician, Computer Scientist, General Expert)
- Gene analysis that combines Open Targets Platform, Open Targets Genetics, and PubMed abstracts (with LLM extraction), plus CSV download of the gene table
- PDF upload with FAISS-based question answering and summary (FAISS is a vector index for searching document chunks)
- A tab that tests connectivity to the Open Targets Platform and Genetics APIs

`chainlit_app.py` is a separate chat-style version of the gene workflow built with Chainlit (`chainlit.md` is its welcome page).

## Requirements

- Python 3 with the packages in `requirements.txt` (requests, chainlit, openai, langchain, langchain_community, langchain_openai, biopython, xmltodict, pypdf, faiss-cpu)
- For the Streamlit app, also `streamlit`, `pandas`, and `toml` (not listed in `requirements.txt`)
- An OpenAI API key

## Setup

```bash
pip install -r requirements.txt
pip install streamlit pandas toml   # for getGPT.py
```

For the Streamlit app, add the key to `.streamlit/secrets.toml`:

```toml
OPENAI_API_KEY = "YOUR_OPENAI_API_KEY"
```

Both apps set `Entrez.email = "your.email@example.com"`. Change it to your own address in the code before heavy PubMed use (NCBI asks for a real contact email).

## Usage

Streamlit app:

```bash
streamlit run getGPT.py
```

Chainlit app:

```bash
chainlit run chainlit_app.py
```

## Security

`chainlit_app.py` currently **hardcodes an OpenAI API key** in source and writes it into `OPENAI_API_KEY`, which overrides any key in your environment. Treat that key as exposed: **rotate it**, delete that line, and set `OPENAI_API_KEY` as an environment variable instead. This README does not include the key.
