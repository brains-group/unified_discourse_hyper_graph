import argparse
import json
import logging
import os
from pathlib import Path
from typing import List

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from src.hipporag.HippoRAG import HippoRAG
from src.hipporag.StandardRAG import StandardRAG
from src.hipporag.utils.config_utils import BaseConfig
from src.hipporag.utils.misc_utils import string_to_bool

PROJECT_ROOT = Path(__file__).resolve().parent
# [unh_v1] datasets live in <repo>/evaluation_data/<dataset>/ (byte-identical to the HippoRAG 2 release)
DATA_ROOT = PROJECT_ROOT.parents[1] / "evaluation_data"


def get_gold_docs(samples: List, dataset_name: str = None) -> List:
    gold_docs = []
    for sample in samples:
        if "supporting_facts" in sample:
            gold_titles = {item[0] for item in sample["supporting_facts"]}
            supporting_contexts = [item for item in sample["context"] if item[0] in gold_titles]
            separator = "" if (dataset_name or "").startswith("hotpotqa") else " "
            gold_doc = [item[0] + "\n" + separator.join(item[1]) for item in supporting_contexts]
        elif "contexts" in sample:
            gold_doc = [item["title"] + "\n" + item["text"] for item in sample["contexts"] if item["is_supporting"]]
        else:
            if "paragraphs" not in sample:
                raise ValueError("Each sample needs supporting_facts, contexts, or paragraphs for retrieval evaluation")
            paragraphs = [item for item in sample["paragraphs"] if item.get("is_supporting", True)]
            gold_doc = [item["title"] + "\n" + (item["text"] if "text" in item else item["paragraph_text"]) for item in paragraphs]
        gold_docs.append(list(dict.fromkeys(gold_doc)))
    return gold_docs


def get_gold_answers(samples):
    gold_answers = []
    for sample in samples:
        if "answer" in sample or "gold_ans" in sample:
            gold_answer = sample.get("answer", sample.get("gold_ans"))
        elif "reference" in sample:
            gold_answer = sample["reference"]
        elif "obj" in sample:
            gold_answer = [sample["obj"], sample["o_wiki_title"]]
            for field in ("possible_answers", "o_aliases"):
                value = sample.get(field, [])
                gold_answer.extend(value if isinstance(value, list) else [value])
        else:
            raise ValueError("Each query sample must contain an answer field")
        if gold_answer is None:
            raise ValueError("Answer fields must not be null")
        answers = [gold_answer] if isinstance(gold_answer, str) else list(gold_answer)
        answers.extend(sample.get("answer_aliases", []))
        gold_answers.append(list(dict.fromkeys(answers)))
    return gold_answers


def parse_args():
    parser = argparse.ArgumentParser(description="HippoRAG retrieval and QA experiments")
    parser.add_argument("--dataset", default="musique", help="Dataset name under evaluation_data/ (hotpotqa, musique)")
    parser.add_argument("--rag_type", choices=["hipporag", "standard"], default="hipporag", help="Retrieval method; standard reproduces the DPR-style dense baseline")
    parser.add_argument("--llm_base_url", default="https://api.openai.com/v1", help="OpenAI-compatible LLM base URL")
    parser.add_argument("--llm_name", default="gpt-4o-mini", help="LLM model name")
    parser.add_argument("--embedding_name", default="nvidia/NV-Embed-v2", help="Embedding model name")
    parser.add_argument("--embedding_provider", choices=["openai", "transformers", "vllm", "gritlm", "nvembed", "contriever", "cohere"], help="Explicit embedding provider; otherwise inferred from the model name")
    parser.add_argument("--azure_endpoint", help="Azure OpenAI resource endpoint")
    parser.add_argument("--azure_api_version", help="Azure OpenAI API version for chat completions")
    parser.add_argument("--azure_chat_deployment", help="Azure OpenAI chat deployment name; defaults to --llm_name")
    parser.add_argument("--azure_embedding_endpoint", help="Azure OpenAI resource endpoint for embeddings")
    parser.add_argument("--azure_embedding_api_version", help="Azure OpenAI embeddings API version; defaults to --azure_api_version")
    parser.add_argument("--azure_embedding_deployment", help="Azure OpenAI embedding deployment name; defaults to --embedding_name")
    # [unh_v1] BaseConfig.embedding_base_url exists upstream but was not exposed on the CLI.
    parser.add_argument("--embedding_base_url", default=None, help="Full embeddings URL for --embedding_provider vllm, e.g. http://localhost:8001/v1/embeddings")
    # [unh_v1] QuALITY: --dataset quality reads <quality_inputs>/{chunks,questions}.json (evaluation_data/build_quality_inputs.py)
    parser.add_argument("--quality_inputs", default=str(DATA_ROOT / "quality" / "inputs"), help="QuALITY inputs dir (--dataset quality)")
    parser.add_argument("--data_root", default=str(DATA_ROOT), help="Directory containing <dataset>/<dataset>.json and <dataset>/<dataset>_corpus.json")
    parser.add_argument("--embedding_batch_size", type=int, default=8, help="Embedding batch size")
    parser.add_argument("--force_index_from_scratch", default="false", help="Rebuild graph state while reusing compatible stored embeddings/OpenIE")
    parser.add_argument("--force_openie_from_scratch", default="false", help="Regenerate OpenIE only in a fresh/empty index directory")
    # [unh_v1] concurrent OpenIE requests; default = BaseConfig's upstream default (8). Speed only.
    parser.add_argument("--openie_max_workers", type=int, default=8, help="Concurrent online OpenIE requests (upstream default 8)")
    parser.add_argument("--openie_mode", choices=["online", "offline", "Transformers-offline"], default="online", help="OpenIE execution mode")
    parser.add_argument("--save_dir", default="outputs", help="Output directory prefix; custom values retain the legacy _<dataset> suffix")
    return parser.parse_args()


def main():
    args = parse_args()
    if Path(args.dataset).name != args.dataset:
        raise ValueError("dataset must be a name, not a path")
    save_dir = str(PROJECT_ROOT / "outputs" / args.dataset) if args.save_dir == "outputs" else f"{args.save_dir}_{args.dataset}"
    quality = args.dataset == "quality"
    if quality:
        # [unh_v1] QuALITY: HippoRAG 2 has no chunker (one document = one passage), so it indexes the 1200/100-token
        # chunks HyGRAG's own chunker makes from each "title\ntext" article; questions carry their four options.
        with open(Path(args.quality_inputs) / "chunks.json", encoding="utf-8") as file:
            corpus = json.load(file)
        with open(Path(args.quality_inputs) / "questions.json", encoding="utf-8") as file:
            samples = json.load(file)
        docs = [chunk["text"] for chunk in corpus]
    else:
        dataset_dir = Path(args.data_root) / args.dataset
        with open(dataset_dir / f"{args.dataset}_corpus.json", encoding="utf-8") as file:
            corpus = json.load(file)
        with open(dataset_dir / f"{args.dataset}.json", encoding="utf-8") as file:
            samples = json.load(file)
        docs = [f"{doc['title']}\n{doc['text']}" for doc in corpus]
    queries = [sample["question"] for sample in samples]
    gold_answers = get_gold_answers(samples)
    try:
        gold_docs = None if quality else get_gold_docs(samples, args.dataset)
    except (ValueError, KeyError):
        logging.warning("Retrieval evaluation is disabled because supporting documents are unavailable")
        gold_docs = None

    config = BaseConfig(
        # [unh_v1] temperature: LLM_TEMPERATURE if set, else not sent (provider default; upstream default is 0)
        temperature=float(os.environ["LLM_TEMPERATURE"]) if os.environ.get("LLM_TEMPERATURE") else None,
        save_dir=save_dir,
        llm_base_url=args.llm_base_url,
        llm_name=args.llm_name,
        azure_endpoint=args.azure_endpoint,
        azure_api_version=args.azure_api_version,
        azure_chat_deployment=args.azure_chat_deployment,
        azure_embedding_endpoint=args.azure_embedding_endpoint,
        azure_embedding_api_version=args.azure_embedding_api_version,
        azure_embedding_deployment=args.azure_embedding_deployment,
        dataset=args.dataset,
        embedding_model_name=args.embedding_name,
        embedding_provider=args.embedding_provider,
        embedding_base_url=args.embedding_base_url,
        force_index_from_scratch=string_to_bool(args.force_index_from_scratch),
        force_openie_from_scratch=string_to_bool(args.force_openie_from_scratch),
        rerank_dspy_file_path=str(PROJECT_ROOT / "src" / "hipporag" / "prompts" / "dspy_prompts" / "filter_llama3.3-70B-Instruct.json"),
        retrieval_top_k=200,
        linking_top_k=5,
        qa_top_k=5,
        # [unh_v1] entity-extraction (NER) reply cap = 2048, as in the paper-era code (Feb 2025), where NER used the general
        # max_new_tokens=2048. The 512 default was added upstream in Aug 2026 (commit 2f52a86). Triple cap stays 2048.
        openie_ner_max_tokens=2048,
        embedding_batch_size=args.embedding_batch_size,
        openie_mode=args.openie_mode,
        openie_max_workers=args.openie_max_workers,
    )
    logging.basicConfig(level=logging.INFO)
    rag_class = HippoRAG if args.rag_type == "hipporag" else StandardRAG
    with rag_class(global_config=config) as rag:
        rag.index(docs)
        if quality:
            # [unh_v1] QuALITY: retrieval only (the same rag.retrieve rag_qa runs first); the answer comes from
            # common/answer.py's multiple-choice prompt, so HippoRAG's own short-answer QA call is skipped.
            solutions = rag.retrieve(queries=queries)
            # [unh_v1] gold = the option letter (QuALITY) or the answer text (G-Medical / contracts inputs, same layout)
            predictions = [{"id": s["id"], "question": sol.question, "gold_answers": [s.get("answer_letter", s.get("answer"))], "pred_answer": "",
                            "raw_response": "", "retrieved_docs": list(sol.docs[:config.qa_top_k]), "gold_docs": None}
                           for s, sol in zip(samples, solutions)]
            os.makedirs(save_dir, exist_ok=True)
            with open(os.path.join(save_dir, "predictions.json"), "w", encoding="utf-8") as file:
                json.dump(predictions, file, ensure_ascii=False, indent=2)
            logging.info("Saved %d retrievals to %s", len(predictions), os.path.join(save_dir, "predictions.json"))
            return
        outputs = rag.rag_qa(queries=queries, gold_docs=gold_docs, gold_answers=gold_answers)

    # [unh_v1] Persist per-question predictions + upstream metrics so evaluation_scripts/common/score.py
    # can add Contain-Acc / LLM-Acc with the same code used for every other system.
    solutions, responses, _metadata = outputs[0], outputs[1], outputs[2]
    retrieval_metrics = outputs[3] if len(outputs) > 3 else None
    qa_metrics = outputs[4] if len(outputs) > 4 else None
    predictions = []
    for sample, solution, response, answers in zip(samples, solutions, responses, gold_answers):
        predictions.append({
            "id": sample.get("id", sample.get("_id")),
            "question": solution.question,
            "gold_answers": answers,
            "pred_answer": solution.answer,
            "raw_response": response,
            "retrieved_docs": list(solution.docs[:config.qa_top_k]),
            "gold_docs": solution.gold_docs,
        })
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, "predictions.json"), "w", encoding="utf-8") as file:
        json.dump(predictions, file, ensure_ascii=False, indent=2)
    with open(os.path.join(save_dir, "upstream_metrics.json"), "w", encoding="utf-8") as file:
        json.dump({"retrieval": retrieval_metrics, "qa": qa_metrics, "rag_type": args.rag_type,
                   "llm_name": args.llm_name, "embedding_name": args.embedding_name}, file, indent=2)
    logging.info("Saved %d predictions to %s", len(predictions), os.path.join(save_dir, "predictions.json"))


if __name__ == "__main__":
    main()
