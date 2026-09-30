from dataclasses import dataclass

from liteparse import LiteParse


@dataclass(frozen=True)
class Page:
    number: int  # 1-based page number
    text: str


class DocumentParser:
    """Extracts layout-preserving text from PDFs, page by page, with LiteParse.

    LiteParse runs locally (a Node.js CLI) and places text by its coordinates on
    the page, so table rows and columns stay aligned. Blocks that sit side by side
    (two text columns, a caption beside a paragraph) come out interleaved line by
    line. With ``ocr_enabled``, text inside images (scanned pages, figures) is
    recovered with Tesseract OCR.
    """

    def __init__(self, ocr_enabled: bool = True, timeout_s: float = 300.0, cli_path: str | None = None):
        # Never let the server run `npm install -g` on its own; fail loudly instead.
        self._parser = LiteParse(cli_path=cli_path, install_if_not_available=False)
        self.ocr_enabled = ocr_enabled
        self.timeout_s = timeout_s

    def parse(self, pdf_bytes: bytes) -> list[Page]:
        """Parses an in-memory PDF (streamed to the CLI over stdin, no temp files).

        Raises liteparse.ParseError for unreadable files, TimeoutError when parsing
        exceeds ``timeout_s`` and liteparse.CLINotFoundError when the CLI is missing.
        """
        result = self._parser.parse(pdf_bytes, ocr_enabled=self.ocr_enabled, timeout=self.timeout_s)
        return [Page(number=page.pageNum, text=page.text) for page in result.pages]
