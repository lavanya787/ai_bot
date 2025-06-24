import pytesseract
from PIL import Image
import io

def ocr_utils(file):
    """
    Extract text from an image file using OCR.
    Args:
        file: The image file object (e.g., PNG, JPG).
    Returns:
        str: The extracted text.
    """
    try:
        # Open the image file
        image = Image.open(io.BytesIO(file.read()))
        
        # Perform OCR using pytesseract
        text = pytesseract.image_to_string(image)
        return text.strip()
    except Exception as e:
        raise Exception(f"OCR failed: {str(e)}")