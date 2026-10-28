from pathlib import Path
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions,
    TableStructureOptions,
    TableFormerMode,
    EasyOcrOptions,
    LayoutOptions,
    CodeFormulaVlmOptions,
    PictureDescriptionApiOptions
)
from docling.datamodel.layout_model_specs import DOCLING_LAYOUT_EGRET_LARGE
from docling.backend.docling_parse_backend import DoclingParseDocumentBackend
from docling.document_converter import DocumentConverter, PdfFormatOption

# Import the PictureItem class so we can find images in the document
from docling_core.types.doc import PictureItem


def pdf_to_txt_max_accuracy(pdf_path: str, output_txt_path: str = None) -> str:
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    if output_txt_path is None:
        output_txt_path = pdf_path.with_suffix(".txt")
    output_txt_path = Path(output_txt_path)

    pipeline_options = PdfPipelineOptions(
        do_ocr=True,
        do_table_structure=True,
        do_code_enrichment=True,
        do_formula_enrichment=True,

        generate_page_images=False,
        generate_picture_images=True,
        images_scale=2.0,
        document_timeout=300.0,
        ocr_batch_size=8,
        layout_batch_size=8,
    )

    pipeline_options.table_structure_options = TableStructureOptions(
        do_cell_matching=True,
        mode=TableFormerMode.ACCURATE,
    )

    pipeline_options.layout_options = LayoutOptions(
        model_spec=DOCLING_LAYOUT_EGRET_LARGE,
        create_orphan_clusters=True,
    )

    pipeline_options.ocr_options = EasyOcrOptions(
        lang=["en"],
        confidence_threshold=0.5,
    )

    pipeline_options.code_formula_options = CodeFormulaVlmOptions.from_preset(
        "codeformulav2"
    )

    # 1. Enable VLM and Remote Services
    pipeline_options.generate_picture_images = True
    pipeline_options.do_picture_description = True
    pipeline_options.enable_remote_services = True

    # 2. FIX: Connect to vLLM with lower token limits to avoid silent 400 Errors
    pipeline_options.picture_description_options = PictureDescriptionApiOptions(
        url="http://localhost:8000/v1/chat/completions",
        params={
            "model": "google/gemma-4-26B-A4B-it",
            "max_tokens": 500,  # LOWERED to prevent context window overflow
            "max_completion_tokens": 500,  # Added for newer vLLM compatibility
        },
        prompt="Describe the image in one to two sentences. Be concise and accurate.",
        timeout=90,
    )

    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                backend=DoclingParseDocumentBackend,
                pipeline_options=pipeline_options,
            )
        }
    )

    print(f"Converting: {pdf_path.name} ...")
    result = converter.convert(str(pdf_path))
    doc = result.document

    # 3. FIX: Export to Markdown with a custom placeholder
    markdown_text = doc.export_to_markdown(image_placeholder="%%IMAGE_PLACEHOLDER%%")

    # 4. FIX: Find the VLM descriptions and inject them into the Markdown
    for element, _ in doc.iterate_items():
        if isinstance(element, PictureItem):

            # Extract the text from Docling's annotation objects
            description_text = ""
            if element.annotations:
                for ann in element.annotations:
                    if hasattr(ann, 'text'):
                        description_text += ann.text + " "
                    elif isinstance(ann, dict) and 'text' in ann:
                        description_text += ann['text'] + " "
                    else:
                        description_text += str(ann) + " "

            # Format the output block
            if not description_text.strip():
                formatted_desc = "\n\n*[Image: No description generated]*\n\n"
            else:
                formatted_desc = f"\n\n*[Image Description: {description_text.strip()}]*\n\n"

            # Replace the first instance of the placeholder sequentially
            markdown_text = markdown_text.replace("%%IMAGE_PLACEHOLDER%%", formatted_desc, 1)

    # Clean up any leftover placeholders if extraction missed them
    markdown_text = markdown_text.replace("%%IMAGE_PLACEHOLDER%%", "\n*[Image]*\n")

    output_txt_path.write_text(markdown_text, encoding="utf-8")
    print(f"Saved to: {output_txt_path}")

    return markdown_text


if __name__ == "__main__":
    pdf_to_txt_max_accuracy(pdf_path="td.pdf", output_txt_path="td_test.txt")