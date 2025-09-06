import pandas as pd
from docling.document_converter import DocumentConverter
from pathlib import Path
import time
import logging

def main():
    logging.basicConfig(level=logging.INFO)
    
    # Path to input document (PDF)
    input_doc_path = r"C:\Users\Pc\Desktop\axe_docl\batch1-0001.jpg"
    output_dir = Path("scratch_table")
    
    # Create document converter instance
    doc_converter = DocumentConverter()
    
    start_time = time.time()
    
    # Convert document
    conv_res = doc_converter.convert(input_doc_path)
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    doc_filename = Path(input_doc_path).stem
    
    # Export tables to CSV files
    for table_ix, table in enumerate(conv_res.document.tables):
        # Convert table to pandas DataFrame
        table_df: pd.DataFrame = table.export_to_dataframe()
        
        # Save as CSV
        output_path = output_dir / f"{doc_filename}_table_{table_ix+1}.csv"
        table_df.to_csv(output_path, index=False)
        logging.info(f"Saved table {table_ix+1} to {output_path}")
    
    elapsed = time.time() - start_time
    logging.info(f"Processed document in {elapsed:.2f} seconds")

if __name__ == "__main__":
    main()