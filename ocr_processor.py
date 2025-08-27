# ocr_processor.py

import os
import pytesseract
from PIL import Image
import fitz  # PyMuPDF
from pathlib import Path
import time

# --- Configuration ---
# Define the paths to your data folders using pathlib for cross-platform compatibility
BASE_DIR = Path(__file__).resolve().parent
PDF_SOURCE_DIR = BASE_DIR / "data" / "raw_pdfs"
TEXT_OUTPUT_DIR = BASE_DIR / "data" / "processed_text"

def process_pdfs():
    """
    Processes all PDF files in the source directory, performs OCR,
    and saves the extracted text to the output directory.
    """
    print("--- Starting PDF Processing ---")

    # Create the output directory if it doesn't exist
    TEXT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Get a list of all PDF files in the source directory
    pdf_files = list(PDF_SOURCE_DIR.glob("*.pdf"))

    if not pdf_files:
        print(f"No PDF files found in {PDF_SOURCE_DIR}")
        print("Please add your scanned PDF reports to the 'data/raw_pdfs' folder.")
        return

    print(f"Found {len(pdf_files)} PDF(s) to process.")

    for pdf_path in pdf_files:
        file_name = pdf_path.stem
        output_txt_path = TEXT_OUTPUT_DIR / f"{file_name}.txt"

        # Check if the text file already exists to avoid reprocessing
        if output_txt_path.exists():
            print(f"Skipping '{pdf_path.name}', text file already exists.")
            continue

        print(f"\nProcessing '{pdf_path.name}'...")
        start_time = time.time()

        try:
            # Open the PDF file
            pdf_document = fitz.open(pdf_path)
            full_text = ""

            # Iterate through each page of the PDF
            for page_num in range(len(pdf_document)):
                page = pdf_document.load_page(page_num)

                # Render the page to an image (pixmap)
                # The higher the DPI, the better the OCR quality, but the slower the process
                pix = page.get_pixmap(dpi=300)

                # Convert the pixmap to a PIL Image
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

                # Use Tesseract to extract text from the image
                # We specify English as the language
                try:
                    text = pytesseract.image_to_string(img, lang='eng')
                    full_text += f"--- Page {page_num + 1} ---\n{text}\n\n"
                except pytesseract.TesseractNotFoundError:
                    print("\n[ERROR] Tesseract is not installed or not in your PATH.")
                    print("Please install Tesseract using 'brew install tesseract' on your Mac.")
                    return

                # Print progress for large documents
                print(f"  - Processed page {page_num + 1}/{len(pdf_document)}")

            # Save the extracted text to a .txt file
            with open(output_txt_path, "w", encoding="utf-8") as text_file:
                text_file.write(full_text)

            pdf_document.close()
            end_time = time.time()
            print(f"Successfully processed '{pdf_path.name}' in {end_time - start_time:.2f} seconds.")

        except Exception as e:
            print(f"[ERROR] Could not process {pdf_path.name}. Reason: {e}")

    print("\n--- PDF Processing Complete ---")
    print(f"All extracted text has been saved in: {TEXT_OUTPUT_DIR}")


if __name__ == "__main__":
    # This block allows the script to be run directly from the command line
    process_pdfs()