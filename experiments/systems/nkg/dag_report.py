#!/usr/bin/env python3
"""
dag_report.py
=============
Runs the NKG DAG retrieval engine (nkg.dag_retrieval.DAGRetriever) over a
handful of HotpotQA/MuSiQue questions against an already-constructed KG, and
renders a PDF debugging report: one section per question showing the
question/type/gold answer/LLM answer/EM+F1, the *initial* query-decomposition
DAG the planner produced, and the *executed* DAG -- every fork branch the
planner's disambiguation step actually ran, colored/traceable back to its
originating plan node, with any branch that came back empty flagged in red.

This exists because two full eval runs (Qwen local_vllm vs. DeepSeek) on the
same 100 HotpotQA questions showed opposite strengths (one much better at
bridge questions, the other at comparison questions) -- aggregate EM/F1 can't
tell you *why*. Seeing the actual per-question DAG structure and where a fork
came back empty is the only way to find that out.

Usage:
    python3 -m experiments.systems.nkg.dag_report \\
        --run-dir sample_exp/nkg/hotpotqa/run_v1 \\
        --dataset hotpotqa \\
        --questions-file evaluation_data/wiki_data/hotpot_questions_100.jsonl \\
        --provider local_vllm --default-retrieval-args --limit 15 \\
        --out-pdf sample_exp/nkg/hotpotqa/run_v1/dag_report.pdf
"""
import os
import sys
import shutil
import argparse
import traceback
from dataclasses import dataclass, field
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from xml.sax.saxutils import escape

from sentence_transformers import SentenceTransformer, CrossEncoder
from transformers import AutoTokenizer
from dotenv import load_dotenv
from PIL import Image as PILImage

from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, PageBreak, Table, TableStyle,
)

from nkg.utils.config import configure_dspy
from experiments.utils.multihop_utils import (
    SUPPORTED_DATASETS, load_questions_jsonl, score_em, score_f1, call_llm_for_answer, truncate_tokens,
)
from experiments.systems.nkg.hqa_musique_evaluation import (
    PROVIDER_DEFAULTS, resolve_provider, dspy_model_string, build_retriever,
    default_retrieval_args, CONSTRUCTION_RESULTS_FILE, KG_FILENAME,
)
from experiments.systems.nkg.dag_diagrams import render_plan_dag, render_execution_dag


# =============================================================================
# PER-QUESTION RECORD
# =============================================================================
@dataclass
class QuestionReport:
    index: int
    question_id: str
    question: str
    qtype: str
    qlevel: str
    gold: str
    aliases: list
    llm_answer: str = ""
    em: float = 0.0
    f1: float = 0.0
    is_fallback: bool = False
    executions: list = field(default_factory=list)
    plan_png: str = None
    exec_png: str = None
    final_context: str = ""
    context_tokens: int = 0
    status: str = "SUCCESS"
    error: str = ""


def _process_one(q: dict, index: int, retriever, retrieval_args: dict, tokenizer,
                  args, cfg, assets_dir: str) -> QuestionReport:
    qid = q.get("question_id", str(index))
    rec = QuestionReport(
        index=index, question_id=qid, question=q["question"],
        qtype=q.get("type", "unknown"), qlevel=q.get("level", ""),
        gold=q["answer"], aliases=q.get("answer_aliases", []),
    )
    try:
        raw_ctx, trace = retriever.retrieve(query=rec.question, return_raw=True, **retrieval_args)
        trunc_ctx, ctx_tok = truncate_tokens(raw_ctx, tokenizer, args.max_context_tokens)

        llm_ans = call_llm_for_answer(
            model=cfg.model, api_base=cfg.api_base, api_key=cfg.api_key,
            question=rec.question, context=trunc_ctx,
            max_tokens=args.answer_max_tokens, temperature=0.0,
        )

        rec.llm_answer = llm_ans
        rec.em = float(score_em(llm_ans, rec.gold, aliases=rec.aliases, dataset=args.dataset))
        rec.f1, _, _ = score_f1(llm_ans, rec.gold, aliases=rec.aliases, dataset=args.dataset)
        rec.is_fallback = trace.is_fallback
        rec.executions = trace.executions
        rec.final_context = trunc_ctx
        rec.context_tokens = ctx_tok

        stem = os.path.join(assets_dir, f"q{index:03d}_{qid}")
        rec.plan_png = render_plan_dag(trace.by_id, trace.levels, f"{stem}_plan.png",
                                        title=f"Q{index}: Initial DAG Plan")
        rec.exec_png = render_execution_dag(trace.executions, f"{stem}_exec.png",
                                             title=f"Q{index}: Executed DAG (fork branches)")
    except Exception as e:
        rec.status = "ERROR"
        rec.error = f"{e}\n{traceback.format_exc()}"
        print(f"[dag_report ERROR] {qid}: {e}")
    return rec


# =============================================================================
# PDF ASSEMBLY
# =============================================================================
def _fit_image(path: str, max_w_in: float, max_h_in: float):
    with PILImage.open(path) as im:
        w_px, h_px = im.size
    w_in, h_in = w_px / 170.0, h_px / 170.0
    scale = min(max_w_in / w_in, max_h_in / h_in, 1.0)
    return RLImage(path, width=w_in * scale * inch, height=h_in * scale * inch)


def _score_color_hex(em: float, f1: float) -> str:
    if em >= 1.0:
        return "#166534"
    if f1 >= 0.5:
        return "#b45309"
    return "#b91c1c"


def _styles():
    ss = getSampleStyleSheet()
    ss.add(ParagraphStyle("QTitle", parent=ss["Heading1"], fontSize=15, spaceAfter=6))
    ss.add(ParagraphStyle("Section", parent=ss["Heading2"], fontSize=11.5,
                           spaceBefore=12, spaceAfter=4, textColor=colors.HexColor("#1f2937")))
    ss.add(ParagraphStyle("Body", parent=ss["Normal"], fontSize=9.5, leading=13))
    ss.add(ParagraphStyle("Mono", parent=ss["Normal"], fontName="Courier", fontSize=8, leading=11))
    ss.add(ParagraphStyle("Badge", parent=ss["Normal"], fontSize=9, textColor=colors.white))
    return ss


def _branch_detail_table(executions: list, ss):
    rows = [["Branch", "Level", "Parents", "Refined query", "Result"]]
    for ex in sorted(executions, key=lambda e: (e.level, e.branch_id)):
        empty = (ex.context_string or "").strip() in ("", "(No relevant information found.)")
        n_paths = len(ex.raw_paths or [])
        result = "no results" if empty else f"{n_paths} path(s)"
        rows.append([
            Paragraph(escape(ex.branch_id), ss["Body"]),
            str(ex.level),
            Paragraph(escape(", ".join(ex.parent_branch_ids) or "—"), ss["Body"]),
            Paragraph(escape(ex.refined_query), ss["Body"]),
            Paragraph(escape(result), ss["Body"]),
        ])
    tbl = Table(rows, colWidths=[0.85 * inch, 0.45 * inch, 1.0 * inch, 3.1 * inch, 1.0 * inch], repeatRows=1)
    tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e5e7eb")),
        ("FONTSIZE", (0, 0), (-1, 0), 8.5),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#d1d5db")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    return tbl


def _question_section(rec: QuestionReport, ss) -> list:
    flow = [PageBreak()]
    badge = f'{rec.qtype}' + (f' / {rec.qlevel}' if rec.qlevel else '')
    flow.append(Paragraph(f"Q{rec.index}  <font size=10 color='#6b7280'>[{escape(badge)}]  {escape(rec.question_id)}</font>", ss["QTitle"]))
    flow.append(Paragraph(escape(rec.question), ss["Body"]))
    flow.append(Spacer(1, 6))

    if rec.status == "ERROR":
        flow.append(Paragraph("ERROR during retrieval/generation:", ss["Section"]))
        flow.append(Paragraph(escape(rec.error).replace("\n", "<br/>"), ss["Mono"]))
        return flow

    color_hex = _score_color_hex(rec.em, rec.f1)
    aliases_str = f" (aliases: {', '.join(rec.aliases)})" if rec.aliases else ""
    info_rows = [
        ["Gold answer", Paragraph(escape(rec.gold) + escape(aliases_str), ss["Body"])],
        ["LLM answer", Paragraph(escape(rec.llm_answer), ss["Body"])],
        ["EM / F1", Paragraph(f"<font color='{color_hex}'><b>{rec.em:.2f} / {rec.f1:.2f}</b></font>", ss["Body"])],
        ["Fallback mode", Paragraph(
            "single-node plan — delegated straight to the wrapped engine (no DAG structure)"
            if rec.is_fallback else "no", ss["Body"])],
    ]
    tbl = Table(info_rows, colWidths=[1.2 * inch, 5.3 * inch])
    tbl.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9.5),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    flow.append(tbl)
    flow.append(Spacer(1, 8))

    flow.append(Paragraph("Initial DAG Plan", ss["Section"]))
    flow.append(_fit_image(rec.plan_png, max_w_in=6.5, max_h_in=6.5))
    flow.append(Spacer(1, 6))

    flow.append(Paragraph("Executed DAG (all fork branches)", ss["Section"]))
    flow.append(_fit_image(rec.exec_png, max_w_in=6.5, max_h_in=8.5))
    flow.append(Spacer(1, 6))

    flow.append(Paragraph("Branch detail", ss["Section"]))
    flow.append(_branch_detail_table(rec.executions, ss))
    flow.append(Spacer(1, 8))

    flow.append(Paragraph(f"Final context fed to generation ({rec.context_tokens} tokens)", ss["Section"]))
    ctx = rec.final_context[:4000] + (" …[truncated for report]" if len(rec.final_context) > 4000 else "")
    flow.append(Paragraph(escape(ctx).replace("\n", "<br/>"), ss["Mono"]))
    return flow


def build_pdf(records: list, args, retrieval_args: dict, cfg, out_path: str):
    ss = _styles()
    doc = SimpleDocTemplate(out_path, pagesize=letter,
                             leftMargin=0.6 * inch, rightMargin=0.6 * inch,
                             topMargin=0.6 * inch, bottomMargin=0.6 * inch)
    flow = []
    ok = [r for r in records if r.status == "SUCCESS"]
    flow.append(Paragraph("NKG DAG Retrieval Debug Report", ss["QTitle"]))
    meta_lines = [
        f"dataset: {args.dataset}", f"engine: dag", f"provider: {cfg.provider}", f"model: {cfg.model}",
        f"run-dir: {os.path.abspath(args.run_dir)}", f"questions in report: {len(records)}",
        f"retrieval args: {retrieval_args}", f"generated: {datetime.now().isoformat(timespec='seconds')}",
    ]
    if ok:
        n = len(ok)
        mean_em = sum(r.em for r in ok) / n
        mean_f1 = sum(r.f1 for r in ok) / n
        meta_lines.append(f"aggregate over this subset: EM={mean_em:.4f}  F1={mean_f1:.4f}  n={n}")
    for line in meta_lines:
        flow.append(Paragraph(escape(line), ss["Body"]))
    for rec in sorted(records, key=lambda r: r.index):
        flow.extend(_question_section(rec, ss))
    doc.build(flow)


# =============================================================================
# MAIN
# =============================================================================
def main():
    p = argparse.ArgumentParser(description="Render a per-question DAG-retrieval debug PDF.")
    p.add_argument("--run-dir", required=True, help="Contains kg.graphml + construction_results.json.")
    p.add_argument("--dataset", required=True, choices=list(SUPPORTED_DATASETS))
    p.add_argument("--questions-file", required=True)

    p.add_argument("--provider", default=None, choices=list(PROVIDER_DEFAULTS))
    p.add_argument("--model", type=str, default=None)
    p.add_argument("--api-base", type=str, default=None)
    p.add_argument("--api-key", type=str, default="")

    p.add_argument("--retrieval-model", type=str, default="voyageai/voyage-4-nano")
    p.add_argument("--cross-encoder-model", type=str, default="cross-encoder/ms-marco-MiniLM-L6-v2")
    p.add_argument("--tokenizer-model", type=str, default="Qwen/Qwen3-30B-A3B-Instruct-2507-FP8")

    p.add_argument("--default-retrieval-args", action="store_true")
    p.add_argument("--top-k-seeds", type=int, default=None)
    p.add_argument("--max-depth", type=int, default=None)
    p.add_argument("--beam-width", type=int, default=None)
    p.add_argument("--final-top-k", type=int, default=None)
    p.add_argument("--mode", type=str, default=None, choices=["all", "hypergraph", "discourse"])
    p.add_argument("--beam-mmr-lambda", type=float, default=None)
    p.add_argument("--final-mmr-lambda", type=float, default=None)
    p.add_argument("--search-mode", type=str, default=None, choices=["beam_search", "blind_expansion"])
    p.add_argument("--max-fork-branches", type=int, default=None)
    p.add_argument("--parallel-fallback-chain-threshold", type=int, default=None,
                   help="When the executed DAG's longest dependency chain is at or under this "
                        "many nodes, also search the original un-decomposed query directly as an "
                        "extra safety-net branch merged into the final context. 0 disables it.")
    p.add_argument("--deduplicate-paths", action=argparse.BooleanOptionalAction, default=True)

    p.add_argument("--max-context-tokens", type=int, default=2000)
    p.add_argument("--answer-max-tokens", type=int, default=128)
    p.add_argument("--judge-max-tokens", type=int, default=8000)
    p.add_argument("--num-retries", type=int, default=1)

    p.add_argument("--limit", type=int, default=15, help="Max number of questions to include in the report.")
    p.add_argument("--offset", type=int, default=0, help="Skip this many questions before taking --limit.")
    p.add_argument("--question-ids", type=str, default=None,
                    help="Comma-separated question_id list; overrides --limit/--offset if given.")
    p.add_argument("--max-workers", type=int, default=3)
    p.add_argument("--out-pdf", type=str, default=None,
                    help="Defaults to <run-dir>/dag_report.pdf")

    args = p.parse_args()
    load_dotenv()
    cfg = resolve_provider(args)

    results_path = os.path.join(args.run_dir, CONSTRUCTION_RESULTS_FILE)
    kg_path = os.path.join(args.run_dir, KG_FILENAME)
    if not os.path.exists(results_path) or not os.path.exists(kg_path):
        print(f"[FATAL] {CONSTRUCTION_RESULTS_FILE} / {KG_FILENAME} not found in {args.run_dir}.")
        sys.exit(1)

    questions = load_questions_jsonl(args.questions_file)
    if args.question_ids:
        wanted = set(args.question_ids.split(","))
        selected = [(i, q) for i, q in enumerate(questions) if q.get("question_id") in wanted]
    else:
        selected = list(enumerate(questions))[args.offset: args.offset + args.limit]

    if not selected:
        print("[FATAL] No questions selected.")
        sys.exit(1)

    def _val(name, default):
        v = getattr(args, name)
        return default if v is None else v

    retrieval_args = default_retrieval_args("dag") if args.default_retrieval_args else {
        "top_k_seeds": _val("top_k_seeds", 9), "max_depth": _val("max_depth", 1),
        "beam_width": _val("beam_width", 6), "final_top_k": _val("final_top_k", 20),
        "mode": _val("mode", "all"), "beam_mmr_lambda": _val("beam_mmr_lambda", 0.6),
        "final_mmr_lambda": _val("final_mmr_lambda", 0.6), "search_mode": _val("search_mode", "beam_search"),
        "max_fork_branches": _val("max_fork_branches", 6),
        "parallel_fallback_chain_threshold": _val("parallel_fallback_chain_threshold", 2),
    }
    retrieval_args["deduplicate_paths"] = args.deduplicate_paths
    retrieval_args["neutral_layout"] = True
    for name, key in [("top_k_seeds", "top_k_seeds"), ("max_depth", "max_depth"),
                       ("beam_width", "beam_width"), ("final_top_k", "final_top_k"),
                       ("mode", "mode"), ("beam_mmr_lambda", "beam_mmr_lambda"),
                       ("final_mmr_lambda", "final_mmr_lambda"), ("search_mode", "search_mode"),
                       ("max_fork_branches", "max_fork_branches"),
                       ("parallel_fallback_chain_threshold", "parallel_fallback_chain_threshold")]:
        val = getattr(args, name)
        if val is not None and args.default_retrieval_args:
            retrieval_args[key] = val

    out_pdf = args.out_pdf or os.path.join(args.run_dir, "dag_report.pdf")
    assets_dir = os.path.splitext(out_pdf)[0] + "_assets"
    if os.path.exists(assets_dir):
        shutil.rmtree(assets_dir)
    os.makedirs(assets_dir, exist_ok=True)

    print(f"[dag_report] {len(selected)} questions, engine=dag, retrieval={retrieval_args}")
    print(f"[dag_report] Loading retrieval models ({args.retrieval_model})...")
    bi_encoder = SentenceTransformer(args.retrieval_model, trust_remote_code=True)
    cross_encoder = CrossEncoder(args.cross_encoder_model)

    print(f"[dag_report] Loading KG from {kg_path}...")
    retriever = build_retriever("dag", bi_encoder, cross_encoder, kg_path)

    configure_dspy(
        model=dspy_model_string(cfg), api_base=cfg.api_base, api_key=cfg.api_key,
        max_tokens=args.judge_max_tokens, num_retries=args.num_retries,
    )

    try:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_model)
    except Exception:
        import tiktoken

        class _TiktokenWrapper:
            def __init__(self, enc): self._enc = enc
            def encode(self, t, **kw): return self._enc.encode(t)
            def decode(self, ids, **kw): return self._enc.decode(ids)

        tokenizer = _TiktokenWrapper(tiktoken.get_encoding("cl100k_base"))

    print(f"[dag_report] Running {len(selected)} questions ({args.max_workers} workers)...")
    records = []
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {
            executor.submit(_process_one, q, i, retriever, retrieval_args, tokenizer, args, cfg, assets_dir): i
            for i, q in selected
        }
        done = 0
        for fut in as_completed(futures):
            records.append(fut.result())
            done += 1
            print(f"[dag_report] {done}/{len(selected)} done")

    print(f"[dag_report] Assembling PDF -> {out_pdf}")
    build_pdf(records, args, retrieval_args, cfg, out_pdf)
    ok = sum(1 for r in records if r.status == "SUCCESS")
    print(f"[dag_report] DONE. {ok}/{len(records)} succeeded. PDF: {out_pdf}  (diagram assets: {assets_dir})")


if __name__ == "__main__":
    main()
