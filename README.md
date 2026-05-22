# Unified Discourse Hypergraph: Rich graph representation combining discourse graph and hypergraph.

This repository contains the implementation of a **Unified Discourse-Hypergraph RAG** framework for graph-based retrieval-augmented generation over complex documents.

The framework converts source documents into a structured graph index and retrieves evidence through graph-based search. It is designed for document collections where answers may depend on entities, conditions, obligations, cross-references, and multi-hop connections across different parts of the text.

## Overview

The framework has two main stages:

1. **Graph Construction**  
   Source documents are processed into a unified graph index by extracting chunks, atomic facts, entities, metadata, and discourse-level links.

2. **Graph-Based Retrieval**  
   User queries are converted into structured retrieval plans. The system retrieves relevant graph paths and evidence, reranks them, and uses the selected context to generate the final response.

<img width="2047" height="1031" alt="image" src="https://github.com/user-attachments/assets/70964198-1b77-4314-bf4b-deaae79a5f91" />



## Installation

Clone the repository:

```bash
git clone <repo-url>
cd <repo-name>
```

Create the environment using Conda:

```bash
conda env create -f environment.yml
conda activate <ENVIRONMENT_NAME>
```

Alternatively, install dependencies using pip:

```bash
python -m venv <ENVIRONMENT_NAME>
source <ENVIRONMENT_NAME>/bin/activate
pip install -r requirements.txt
```
