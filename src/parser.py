"""
Layout-Aware PDF Parser (parser.py)
Extracts raw text from regulatory PDF documents.
Segments text hierarchically (Chapter -> Section -> Article) based on fonts and legal structures,
preserving context metadata and page lineages.
"""

import os
import re
import fitz  # PyMuPDF
from typing import List, Dict, Any, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("PDFParser")


class HierarchicalPDFParser:
    """
    Parses financial regulatory PDFs and chunks them based on their document structure
    (Chapters, Sections, Articles) using visual layout clues (fonts, capitalization).
    """

    def __init__(self):
        # Regex patterns to detect structural elements in legal text
        self.chapter_pattern = re.compile(r"^(CHAPTER|CHAPITRE)\s+[I|V|X|L|C|\d]+", re.IGNORECASE)
        self.section_pattern = re.compile(r"^(SECTION|PARTIE)\s+[I|V|X|L|C|\d]+", re.IGNORECASE)
        self.article_pattern = re.compile(r"^(ARTICLE|Art\.)\s+\d+", re.IGNORECASE)

    def parse_pdf(self, pdf_path: str) -> List[Dict[str, Any]]:
        """
        Parses a PDF file and returns a list of chunks, each retaining its hierarchical lineage.

        Args:
            pdf_path: Absolute path to the PDF file.

        Returns:
            List of dictionaries containing text chunks and operational metadata.
        """
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF not found at: {pdf_path}")

        logger.info(f"Opening PDF: {pdf_path}")
        doc = fitz.open(pdf_path)
        pdf_name = os.path.basename(pdf_path)

        chunks = []
        current_chapter = "General Context"
        current_section = "General Section"
        current_article = "Overview"
        
        # We will collect text blocks belonging to the current article/section
        current_chunk_text = []
        current_chunk_pages = []
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            # Get text blocks with structural information (font size, style)
            blocks = page.get_text("blocks")
            
            # Sort blocks from top to bottom, left to right
            blocks.sort(key=lambda b: (b[1], b[0]))
            
            for block in blocks:
                text = block[4].strip()
                if not text:
                    continue

                # Clean clean line breaks inside block
                cleaned_text = re.sub(r"\s+", " ", text).strip()
                
                # Check for structural changes
                is_structure = False
                
                if self.chapter_pattern.match(cleaned_text):
                    # Save what we have accumulated before changing chapter
                    if current_chunk_text:
                        chunks.append(self._create_chunk(
                            text=" ".join(current_chunk_text),
                            pdf_name=pdf_name,
                            pages=list(set(current_chunk_pages)),
                            chapter=current_chapter,
                            section=current_section,
                            article=current_article
                        ))
                        current_chunk_text = []
                        current_chunk_pages = []
                    
                    current_chapter = cleaned_text
                    current_section = "General Section"
                    current_article = "Introduction"
                    is_structure = True
                    logger.debug(f"Detected Chapter: {current_chapter}")
                    
                elif self.section_pattern.match(cleaned_text):
                    if current_chunk_text:
                        chunks.append(self._create_chunk(
                            text=" ".join(current_chunk_text),
                            pdf_name=pdf_name,
                            pages=list(set(current_chunk_pages)),
                            chapter=current_chapter,
                            section=current_section,
                            article=current_article
                        ))
                        current_chunk_text = []
                        current_chunk_pages = []
                    
                    current_section = cleaned_text
                    current_article = "Introduction"
                    is_structure = True
                    logger.debug(f"Detected Section: {current_section}")
                    
                elif self.article_pattern.match(cleaned_text):
                    if current_chunk_text:
                        chunks.append(self._create_chunk(
                            text=" ".join(current_chunk_text),
                            pdf_name=pdf_name,
                            pages=list(set(current_chunk_pages)),
                            chapter=current_chapter,
                            section=current_section,
                            article=current_article
                        ))
                        current_chunk_text = []
                        current_chunk_pages = []
                    
                    # Capture the full article title line if possible
                    current_article = cleaned_text[:120]  # Cap length to prevent overflow in metadata
                    is_structure = True
                    logger.debug(f"Detected Article: {current_article}")

                if not is_structure:
                    # Accumulate standard text block
                    current_chunk_text.append(cleaned_text)
                    current_chunk_pages.append(page_num + 1)  # 1-indexed pages

        # Append final remaining chunk
        if current_chunk_text:
            chunks.append(self._create_chunk(
                text=" ".join(current_chunk_text),
                pdf_name=pdf_name,
                pages=list(set(current_chunk_pages)),
                chapter=current_chapter,
                section=current_section,
                article=current_article
            ))

        logger.info(f"Parsed {pdf_name}. Generated {len(chunks)} hierarchical chunks.")
        return chunks

    def _create_chunk(
        self,
        text: str,
        pdf_name: str,
        pages: List[int],
        chapter: str,
        section: str,
        article: str
    ) -> Dict[str, Any]:
        """Utility to format extracted data into a structured dictionary."""
        hierarchy = f"{chapter} > {section} > {article}"
        return {
            "text": text,
            "metadata": {
                "pdf_name": pdf_name,
                "pages": pages,
                "chapter": chapter,
                "section": section,
                "article": article,
                "hierarchy_path": hierarchy
            }
        }


if __name__ == "__main__":
    # Test script - expects a PDF file to run (we will write a test dummy pdf or read one)
    parser = HierarchicalPDFParser()
    print("PDF Parser Initialized.")
