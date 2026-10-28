from pathlib import Path
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions,
    TableStructureOptions,
    TableFormerMode,
    EasyOcrOptions,
    LayoutOptions,
    CodeFormulaVlmOptions,
)
from docling.datamodel.layout_model_specs import DOCLING_LAYOUT_HERON
from docling.backend.docling_parse_backend import DoclingParseDocumentBackend
from docling.document_converter import DocumentConverter, PdfFormatOption
from pathlib import Path


def pdf_to_txt(pdf_path: str, output_txt_path: str = None) -> str:
    """
    Converts a PDF file to a clean, structured text file using Docling,
    with all major features enabled: TableFormer (ACCURATE mode), OCR,
    layout analysis (Heron model), code enrichment, and formula enrichment.

    Args:
        pdf_path:        Path to the input PDF file.
        output_txt_path: Optional path for the output .txt file.
                         Defaults to the same directory as the PDF with a .txt extension.

    Returns:
        The full extracted text as a string.
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    if output_txt_path is None:
        output_txt_path = pdf_path.with_suffix(".txt")
    output_txt_path = Path(output_txt_path)

    # -------------------------------------------------------------------------
    # 1. Pipeline Options — enable every major feature
    # -------------------------------------------------------------------------
    pipeline_options = PdfPipelineOptions(
        do_ocr=True,                    # OCR for scanned/image-based pages
        do_table_structure=True,        # TableFormer: detect & reconstruct tables
        do_code_enrichment=True,        # Extract code blocks accurately
        do_formula_enrichment=True,     # Extract mathematical formulas
        generate_page_images=False,     # We don't need page images for txt output
        generate_picture_images=False,  # We don't need figure images for txt output
        images_scale=2.0,               # High resolution if images are ever needed
        document_timeout=500.0,         # Allow 5 min for large/complex PDFs
        ocr_batch_size=8,               # Process OCR in larger batches for speed
        layout_batch_size=8,            # Process layout in larger batches for speed
    )

    # -------------------------------------------------------------------------
    # 2. TableFormer — use ACCURATE mode for best table reconstruction
    # -------------------------------------------------------------------------
    pipeline_options.table_structure_options = TableStructureOptions(
        do_cell_matching=True,          # Match table cells to PDF's embedded text
        mode=TableFormerMode.ACCURATE,  # Slower but significantly better on complex tables
    )

    # -------------------------------------------------------------------------
    # 3. Layout Analysis — use the Heron model (latest, most accurate)
    #    and create orphan clusters so no floating text is lost
    # -------------------------------------------------------------------------
    pipeline_options.layout_options = LayoutOptions(
        model_spec=DOCLING_LAYOUT_HERON,
        create_orphan_clusters=True,    # Captures floating captions, footnotes, headers
    )

    # -------------------------------------------------------------------------
    # 4. OCR — EasyOCR with English support and a reasonable confidence threshold
    # -------------------------------------------------------------------------
    pipeline_options.ocr_options = EasyOcrOptions(
        lang=["en"],
        confidence_threshold=0.5,       # Discard low-confidence OCR guesses
    )

    # -------------------------------------------------------------------------
    # 5. Code & Formula enrichment model — use the v2 preset (most capable)
    # -------------------------------------------------------------------------
    pipeline_options.code_formula_options = CodeFormulaVlmOptions.from_preset(
        "codeformulav2"
    )

    # -------------------------------------------------------------------------
    # 6. Build the converter using DoclingParseDocumentBackend
    #    (the recommended native backend for PDF parsing)
    # -------------------------------------------------------------------------
    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                backend=DoclingParseDocumentBackend,
                pipeline_options=pipeline_options,
            )
        }
    )

    # -------------------------------------------------------------------------
    # 7. Convert and export to Markdown-flavored text
    #    (Markdown is the best plain-text format because it preserves headings,
    #     table structure, lists, and code blocks in a readable way)
    # -------------------------------------------------------------------------
    print(f"Converting: {pdf_path.name} ...")
    result = converter.convert(str(pdf_path))

    # Export to Markdown text — this preserves all structure (tables, headings,
    # code blocks, formulas) in the cleanest readable plain-text format
    markdown_text = result.document.export_to_markdown()

    # Write to output file
    output_txt_path.write_text(markdown_text, encoding="utf-8")
    print(f"Saved to: {output_txt_path}")

    return markdown_text


# -------------------------------------------------------------------------
# Example usage
# -------------------------------------------------------------------------
def convert_directory(directory_path: str):
    """
    Recursively finds all .pdf files in the given directory and its subdirectories.
    Converts each to a .txt file placed in the exact same directory as the source PDF.
    """
    target_dir = Path(directory_path)

    if not target_dir.is_dir():
        print(f"Error: '{directory_path}' is not a valid directory.")
        return

    # rglob("*.pdf") recursively finds all pdfs in the directory and subdirectories
    pdf_files = list(target_dir.rglob("*.pdf"))

    if not pdf_files:
        print(f"No PDF files found in '{directory_path}' or its subdirectories.")
        return

    print(f"Found {len(pdf_files)} PDF(s). Starting batch conversion...\n")

    success_count = 0
    for pdf_file in pdf_files:
        try:
            # We only need to pass the pdf_file path.
            # Your pdf_to_txt function already defaults to placing the .txt
            # in the same directory with the same base name.
            pdf_to_txt(str(pdf_file))
            success_count += 1
        except Exception as e:
            print(f"Failed to convert {pdf_file}: {e}")

    print(f"\nBatch conversion complete. Successfully converted {success_count}/{len(pdf_files)} files.")

if __name__ == "__main__":
    convert_directory("./evaluation_data")

