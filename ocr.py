
import torch
from doctr.io import DocumentFile
from doctr.models import ocr_predictor  # OCR = Optical Character Recognition, extracts text from images or PDFs

# Configuration
TEXT_DETECTION_MODEL = 'fast_base'  # Model for detecting where text is on the page
TEXT_RECOGNITION_MODEL = 'crnn_vgg16_bn'  # Model for recognizing characters within the detected text regions

DEVICE = torch.device("cuda:0") # The OCR must run on a gpu or it will seg fault.

class OCR:
    def __init__(self):
        # Load OCR model with text detection and recognition components
        self.ocr_model = ocr_predictor(det_arch=TEXT_DETECTION_MODEL,
                                       reco_arch=TEXT_RECOGNITION_MODEL,
                                       pretrained=True,
                                       assume_straight_pages=True,
                                       preserve_aspect_ratio=True).to(DEVICE)
        self.ocr_model.doc_builder.resolve_lines = True  # Group words into lines
        self.ocr_model.doc_builder.resolve_blocks = True  # Group lines into blocks (paragraphs)


    def extract_pdf_data(self, path, page_index=None):
        if page_index is not None:
            pages = DocumentFile.from_pdf(path, page_indices=[page_index])
        else:
            pages = DocumentFile.from_pdf(path)
        return self.ocr_model(pages)


    @staticmethod
    def _convert_block_data_to_text(block_data):
        lines = []
        for line in block_data.lines:
            # Join words with spaces and strip trailing whitespace
            line_text = " ".join(word.value for word in line.words).rstrip()
                        
            # Remove trailing hyphen if present
            if line_text.endswith('-'):
                line_text = line_text[:-1].rstrip()  # Remove hyphen and any remaining whitespace
            
            lines.append(line_text)
            
        # Join processed lines with single space
        return " ".join(lines)


    @staticmethod
    def _convert_doc_data_to_text(doc_data):
        """
        Converts an OCR document into plain text.

        Args:
            block_data: A doc object from docTR OCR output.

        Returns:
            str: Text content from the document.
        """
        blocks = []
        for page in doc_data.pages:
            for block in page.blocks:
                block_text = OCR._convert_block_data_to_text(block)
                blocks.append(block_text)
    
        # Separate text blocks within a document
        return "\n\n".join(blocks)