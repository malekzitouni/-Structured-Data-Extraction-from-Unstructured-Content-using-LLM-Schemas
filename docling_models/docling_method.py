import json
import inspect
import warnings
from pathlib import Path
from typing import Any
from docling.datamodel.base_models import InputFormat
from docling_core.types.doc import ImageRefMode
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions,
    TableFormerMode,
    TesseractOcrOptions,
)
from docling.datamodel.document import ConversionResult

def convert_result_to_json(result: Any, output_path: Path) -> None:
    """Safe JSON serialization with warning suppression"""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        json_data = result.model_dump_json(
            indent=2,
            exclude_none=True,
            by_alias=True,
        )
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(json_data)

def main():
    # Configure output directory
    output_dir = Path("parsed-doc-advanced")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Define pipeline options
    pipeline_options = PdfPipelineOptions(
        do_table_structure=True,
        do_ocr=True,
        ocr_options=TesseractOcrOptions(
            force_full_page_ocr=True,
            lang=["eng"]
        ),
        table_structure_options=dict(
            do_cell_matching=False,
            mode=TableFormerMode.ACCURATE,
        ),
        generate_page_images=True,
        generate_picture_images=True,
        images_scale=2.0,
    )

    # Initialize converter
    doc_converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )

    # Process the input file
    input_path = Path(r"C:\Users\Pc\Desktop\axe_docl\viz_output\365ba580-56a9-4dff-8d16-26d6be90de56.jpg")
    conv_result = doc_converter.convert(str(input_path))
    # Access layout information from ConversionResult
   
   # Access layout information from ConversionResult
    for page in conv_result.pages:
       print(f"\nPage {page.page_no} dimensions: {page.size}")
    
    # Access parsed content (text and structure)
       if hasattr(page.parsed_page, 'text'):
        print(f"Text content length: {len(page.parsed_page.text)} characters")
    
    # Access tables if available
       if hasattr(page.parsed_page, 'tables'):
        print(f"Found {len(page.parsed_page.tables)} tables:")
        for i, table in enumerate(page.parsed_page.tables):
            print(f"  Table {i+1}:")
            print(f"    Position: {table.bbox}")  # (x1, y1, x2, y2)
            print(f"    Rows: {len(table.data)}")
            print(f"    Columns: {len(table.data[0]) if table.data else 0}")
    
    # Access other regions (non-table content)
       if hasattr(page.parsed_page, 'blocks'):  # Some versions use 'blocks'
        print("Document regions:")
        for block in page.parsed_page.blocks:
            print(f"  - Type: {block.type} | Position: {block.bbox}")
            if hasattr(block, 'text'):
                print(f"    Text: {block.text[:50]}...")



    doc_filename = input_path.stem

    # Suppress specific deprecation warnings
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Parameter `strict_text` has been deprecated")
        warnings.filterwarnings("ignore", message="Use export_to_doctags() instead")

        # 1. Export Deep Search JSON format
        deepsearch_path = output_dir / f"{doc_filename}.deepsearch.json"
        with open(deepsearch_path, "w", encoding="utf-8") as fp:
            json.dump(conv_result.document.export_to_dict(), fp, indent=2)
        print(f"Deep Search JSON saved to: {deepsearch_path}")

        # 2. Export Text format
        text_path = output_dir / f"{doc_filename}.txt"
        with open(text_path, "w", encoding="utf-8") as fp:
            fp.write(conv_result.document.export_to_text())
        print(f"Text export saved to: {text_path}")

        # 3. Export Markdown formats
        md_path = output_dir / f"{doc_filename}.md"
        conv_result.document.save_as_markdown(md_path, image_mode=ImageRefMode.REFERENCED)
        
        export_md_path = output_dir / f"{doc_filename}.export.md"
        with open(export_md_path, "w", encoding="utf-8") as fp:
            fp.write(conv_result.document.export_to_markdown())
        print(f"Markdown exports saved to: {md_path} and {export_md_path}")

        # 4. Export Document Tags format (using new method if available)
        doctags_path = output_dir / f"{doc_filename}.doctags"
        with open(doctags_path, "w", encoding="utf-8") as fp:
            if hasattr(conv_result.document, 'export_to_doctags'):
                fp.write(conv_result.document.export_to_doctags())
            else:
                fp.write(conv_result.document.export_to_document_tokens())
        print(f"Document tags saved to: {doctags_path}")

    # 5. Save full ConversionResult as JSON
    json_output_path = output_dir / "result_finale.json"
    convert_result_to_json(conv_result, json_output_path)
    print(f"Full ConversionResult JSON saved to: {json_output_path}")

    # 6. Save as plain text
    text_output_path = output_dir / "result_finale_pa.txt"
    with open(text_output_path, "w", encoding="utf-8") as f:
        f.write(str(conv_result))
    print(f"Plain text report saved to: {text_output_path}")

    print("\nAll export tasks completed successfully!")

if __name__ == "__main__":
    # Print ConversionResult structure for reference
    print(inspect.getsource(ConversionResult))
    
    # Run main processing
    main()