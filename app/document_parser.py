from liteparse import LiteParse

class DocumentParser:
    def __init__(self, chunk_size=1000, chunk_overlap=200):
        # Initialize the LiteParse engine
        self.parser = LiteParse()
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def process_pdf(self, file_path: str) -> list[str]:
        """
        Takes a PDF path, extracts the layout-aware text, 
        and returns a list of overlapping text chunks.
        """
        # Parse the document
        result = self.parser.parse(file_path)
        extracted_text = result.text
        
        # Chunk the text using the sliding window algorithm
        chunks = []
        for i in range(0, len(extracted_text), self.chunk_size - self.chunk_overlap):
            chunk = extracted_text[i : i + self.chunk_size]
            chunks.append(chunk)
            
        return chunks