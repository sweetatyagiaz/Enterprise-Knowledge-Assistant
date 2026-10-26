from io import BytesIO
from pathlib import Path

from docx import Document
from pypdf import PdfReader


SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md"}


def load_document(filename: str, content: bytes) -> str:
    """Extract text from supported document formats."""

    extension = Path(filename).suffix.lower()

    if extension not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file type: {extension}. "
            f"Supported types: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )

    if not content:
        raise ValueError("The uploaded document is empty.")

    if extension in {".txt", ".md"}:
        return content.decode("utf-8-sig")

    if extension == ".pdf":
        try:
            reader = PdfReader(BytesIO(content))

            if reader.is_encrypted:
                raise ValueError("Encrypted PDFs are not supported.")

            text = "\n\n".join(
                page.extract_text() or ""
                for page in reader.pages
            ).strip()
        except ValueError:
            raise
        except Exception as exc:
            raise ValueError("Invalid PDF document.") from exc

        if not text:
            raise ValueError("The PDF contains no usable text.")

        return text

    if extension == ".docx":
        try:
            document = Document(BytesIO(content))
            paragraphs = [
                paragraph.text
                for paragraph in document.paragraphs
                if paragraph.text.strip()
            ]
            text = "\n".join(paragraphs).strip()
        except Exception as exc:
            raise ValueError("Invalid DOCX document.") from exc

        if not text:
            raise ValueError("The DOCX contains no usable text.")

        return text

    raise ValueError(f"Unsupported file type: {extension}")