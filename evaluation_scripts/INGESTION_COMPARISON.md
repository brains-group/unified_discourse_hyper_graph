# How HippoRAG 2, MemGraphRAG and LogicRAG ingest HotpotQA

In short:
- **MemGraphRAG** reads the whole `hotpotqa.txt` as one string and cuts it into token windows that run across passage boundaries.
- **HippoRAG 2** and **LogicRAG** read `hotpotqa_corpus.json` and turn each passage into its own chunk.
- None of the three builds its index from the question JSON's `context` field; all index the shared corpus.

All links are pinned to commits (checked 2026-09-23), so line numbers stay valid.

## MemGraphRAG: reads the whole `hotpotqa.txt` and cuts it into windows across passage boundaries

File: `code/index.py` at commit `cd6fabd`

- **Lines 75–78** ([link](https://github.com/XMUDeepLIT/MemGraphRAG/blob/cd6fabda1ec31a302bb283089e17c087ab7713e8/code/index.py#L75-L78)) are the key part:
  - `corpus = handle.read()` reads the entire file into one string.
  - `chunks = split_text(corpus, …)` cuts that string into windows.
  - `docs = [f"{idx}:{chunk}" …]` numbers the windows.
- **Lines 16–36** ([link](https://github.com/XMUDeepLIT/MemGraphRAG/blob/cd6fabda1ec31a302bb283089e17c087ab7713e8/code/index.py#L16-L36)), `split_text`:
  - Line 24 turns the whole corpus into tokens in one pass.
  - Line 30 steps through it in windows of `chunk_size - overlap`.
  - Nothing looks for the blank line between passages, so windows run straight across them.
- **Line 110** ([link](https://github.com/XMUDeepLIT/MemGraphRAG/blob/cd6fabda1ec31a302bb283089e17c087ab7713e8/code/index.py#L110)) indexes those windows with `index_with_memory(docs)`.
- The default window is 256 tokens with 32 overlap. The README's example command passes `--corpus` a `.txt` file ([README](https://github.com/XMUDeepLIT/MemGraphRAG/blob/cd6fabda1ec31a302bb283089e17c087ab7713e8/README.md), "Run Indexing").

## HippoRAG 2: one chunk per passage, built from `hotpotqa_corpus.json`

File: `main.py` at commit `1438aba`

- **Line 88** ([link](https://github.com/OSU-NLP-Group/HippoRAG/blob/1438aba3fc44ff10573e5a5e1e7cc3c7f9794aff/main.py#L88)) loads `{dataset}_corpus.json`, not a `.txt`.
- **Line 93** ([link](https://github.com/OSU-NLP-Group/HippoRAG/blob/1438aba3fc44ff10573e5a5e1e7cc3c7f9794aff/main.py#L93)) builds `docs = [f"{doc['title']}\n{doc['text']}" for doc in corpus]`, one string per passage.
- **Line 127** ([link](https://github.com/OSU-NLP-Group/HippoRAG/blob/1438aba3fc44ff10573e5a5e1e7cc3c7f9794aff/main.py#L127)) calls `rag.index(docs)`.
- To confirm nothing re-chunks them later:
  - `src/hipporag/preprocessing.py` **lines 15–27** ([link](https://github.com/OSU-NLP-Group/HippoRAG/blob/1438aba3fc44ff10573e5a5e1e7cc3c7f9794aff/src/hipporag/preprocessing.py#L15-L27)): the docstring reads "keeps one chunk per document", and each string becomes exactly one `Chunk`.
  - `HippoRAG.index()` applies that at `src/hipporag/HippoRAG.py` **lines 488–520** ([link](https://github.com/OSU-NLP-Group/HippoRAG/blob/1438aba3fc44ff10573e5a5e1e7cc3c7f9794aff/src/hipporag/HippoRAG.py#L488-L520)).

## LogicRAG: one entry per passage, built from `hotpotqa_corpus.json`

File: `src/models/base_rag.py` at commit `72a298c`

- **Lines 42–51** ([link](https://github.com/chensyCN/LogicRAG/blob/72a298c95ffd75890c7f9c28fd1c094201f669a7/src/models/base_rag.py#L42-L51)):
  - `load_corpus` runs `json.load` on the corpus file.
  - It builds `self.corpus = {i: f"Title: {doc['title']}. Content: {doc['text']}" for i, doc in enumerate(documents)}`, one entry per passage.
- **Lines 66–67** ([link](https://github.com/chensyCN/LogicRAG/blob/72a298c95ffd75890c7f9c28fd1c094201f669a7/src/models/base_rag.py#L66-L67)) embed each passage separately.
- **Lines 149–155** ([link](https://github.com/chensyCN/LogicRAG/blob/72a298c95ffd75890c7f9c28fd1c094201f669a7/src/models/base_rag.py#L149-L155)) do retrieval: cosine similarity, then `topk`, returning whole passages.
- `src/main.py` **line 32** ([link](https://github.com/chensyCN/LogicRAG/blob/72a298c95ffd75890c7f9c28fd1c094201f669a7/src/main.py#L32)) defaults `--corpus` to `dataset/hotpotqa_corpus.json`.

## The data files

- MemGraphRAG's `.txt` input is [`dataset/hotpotqa/hotpotqa.txt`](https://github.com/XMUDeepLIT/MemGraphRAG/blob/cd6fabda1ec31a302bb283089e17c087ab7713e8/dataset/hotpotqa/hotpotqa.txt). It's the same corpus as HippoRAG's [`reproduce/dataset/hotpotqa_corpus.json`](https://github.com/OSU-NLP-Group/HippoRAG/blob/1438aba3fc44ff10573e5a5e1e7cc3c7f9794aff/reproduce/dataset/hotpotqa_corpus.json), written out as `title\ntext` blocks.
- The JSON files are byte-identical across HippoRAG, MemGraphRAG, and the [HuggingFace release](https://huggingface.co/datasets/osunlp/HippoRAG_2).
- Local copies are in [`evaluation_data/hotpotqa/`](../evaluation_data/hotpotqa/).

## Local copies in this repo

- [`memgraphrag/index.py`](memgraphrag/index.py)
- [`hipporag2/main.py`](hipporag2/main.py)
- [`logicrag/src/models/base_rag.py`](logicrag/src/models/base_rag.py)

Some of the local patches (tagged `[unh_v1]`) shift line numbers slightly, so use the GitHub links above for exact lines.
