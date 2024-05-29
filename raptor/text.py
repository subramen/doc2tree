from abc import ABC, abstractmethod
from typing import List, Tuple, Generator, Union, Dict, Optional
import re
from transformers import AutoTokenizer
import concurrent.futures
import fitz
import fire
import json
import logging


def clean(text):
    text = text.encode('ascii', 'ignore').decode('utf-8')
    text = text.replace('-\n', '')
    text = text.replace('\n', ' ')
    text = text.replace('\t', ' ')
    text = text.replace('\r', ' ')
    return text

class BaseChunker(ABC):
    @abstractmethod
    def chunk_spans(self, text: str) -> List[Tuple[int, int]]:
        pass

class SentencePreservingChunker:
    def __init__(self, tokenizer: AutoTokenizer, max_chunk_len: int):
        """
        Initialize the Chunker object with a tokenizer.

        Args:
            tokenizer (AutoTokenizer): A tokenizer object used to tokenize text.
            max_seq_len (int): The maximum number of tokens allowed in a sequence.
        """
        self.tokenizer = tokenizer
        self.max_chunk_len = max_chunk_len

    def sentence_spans(self, text: str) -> List[Tuple[int, int]]:
        """
        Get the sentence boundaries in the given text.

        Args:
            text (str): The text to find sentence boundaries in.
            sentence_max_len (int, optional): The maximum number of characters allowed in a sentence. Defaults to -1.

        Returns:
            list: A list of tuples representing the start and end indices of each sentence in the text.
        """
        pattern = r'(?<=[.!?;])\s'
        sentence_splits = [match.start() for match in re.finditer(pattern, text)]
        sentence_idx = []
        c = 0
        for idx in sentence_splits:
            sentence_idx.append((c, idx,))
            c = idx + 1
        return sentence_idx


    def chunk_spans(self, text: str) -> List[Tuple[int, int]]:
        """
        Get the chunk boundaries in the given text.

        Args:
            text (str): The text to find chunk boundaries in.

        Returns:
            list: A list of tuples representing the start and end indices of each chunk in the text.
        """

        # Get the sentence boundaries in the text
        sentence_idx = self.sentence_spans(text)
        # Initialize empty list to store chunk indices
        chunk_idx = []
        chunk = [0, 0]
        current_length = 0

        # Iterate over the sentence boundaries
        for start_ix, end_ix in sentence_idx:
            seq_len = len(self.tokenizer.encode(text[start_ix:end_ix]))

            if current_length + seq_len < self.max_chunk_len:
                chunk[-1] = end_ix
                current_length += seq_len

            else:
                chunk_idx.append(chunk)
                chunk = [start_ix, end_ix]
                current_length = seq_len

        chunk_idx.append(chunk)
        return chunk_idx

    def chunk(self, text: str, return_spans=False) -> List[str]:
        """
        Get the chunks of the given text.

        Args:
            text (str): The text to find chunk boundaries in.

        Returns:
            list: A list of
        """
        chunk_spans = self.chunk_spans(text)
        for start_ix, end_ix in chunk_spans:
            chunk_text = text[start_ix:end_ix]
            if return_spans:
                yield (chunk_text, (start_ix, end_ix))
            yield chunk_text

class Document:
    def __init__(self, path: str):
        self.document = fitz.open(path)
        self.metadata = {k:v for k,v in self.document.metadata.items() if k in ['title', 'author', 'subject']}
        self.metadata['filepath'] = path
        self.sections = self._document_sections()
        self.page_spans = self._page_spans()

    def _document_sections(self, min_chars=280):
        toc = self.document.get_toc()
        sections = []

        for ix, [level, title, pg] in enumerate(toc):
            buffer = -1 # to account for any page numbers miscalculated by pymupdf
            start_pg = max(0, pg + buffer)
            try:
                end_pg = toc[ix+1][-1] + buffer
            except IndexError:
                end_pg = self.document.page_count

            # validate section
            if start_pg < 0 or end_pg < 0:
                continue
            if start_pg > self.document.page_count or end_pg > self.document.page_count:
                continue
            if start_pg >= end_pg:
                continue
            try:
                if start_pg < sections[-1]['start_page']:
                    continue
            except IndexError:
                pass
            # keep section if text is long enough
            section_text = " ".join([page.get_text() for page in self.document.pages(start_pg, end_pg)])
            if len(section_text) < min_chars:
                continue

            # get breadcrumb for this section
            breadcrumb = [""] * level
            candidates = [s for s in toc[:ix] if s[0] < level]
            for c in candidates:
                breadcrumb[c[0]-1] = c[1]
            breadcrumb[-1] = title
            breadcrumb = " > ".join(breadcrumb)

            sections.append({
                "breadcrumb": breadcrumb,
                "start_page": start_pg,
                "end_page": end_pg,
            })

        return sections

    def _page_spans(self):
        page_spans = [[0, 0]]
        for pg in self.document:
            text = pg.get_text()
            page_spans[-1][-1] = page_spans[-1][0] + len(text) + 1
            page_spans.append([page_spans[-1][-1], page_spans[-1][-1]])
        return page_spans

    def read(self, start_end=(0, None)):
        start_end = (start_end[0], start_end[1] or self.document.page_count)
        return "".join([page.get_text() for page in self.document.pages(*start_end)])

    def search_for_chunk(self, chunk_start_ix):
        page_number, page_start_ix = [(c, span[0]) for c, span in enumerate(self.page_spans) if span[0] <= chunk_start_ix < span[1]][0]
        text_str = self.document[page_number].get_text()[chunk_start_ix - page_start_ix : 50 + chunk_start_ix - page_start_ix]
        bbox = self._get_bbox(page_number, text_str)
        return page_number, bbox

    def _get_bbox(self, page_number, text):
        try:
            bbox = list(self.document[page_number].search_for(text)[0])
        except IndexError:
            logging.info(f"pymupdf could not find text '{text}...' in page {page_number}")
            bbox = []
        return bbox

    def get_breadcrumb(self, page_number):
        return [x['breadcrumb'] for x in reversed(self.sections) if x['start_page'] <= page_number][0]



def get_document_chunks(doc: Document, chunker: BaseChunker, start_end=(0, None)):
    text = doc.read(start_end)
    chunk_spans = chunker.chunk_spans(text)
    start_span = doc.page_spans[start_end[0]][0]

    def process_chunk(chunk_start, chunk_end):
        chunk = clean(text[chunk_start:chunk_end])
        chunk_page, bbox = doc.search_for_chunk(start_span + chunk_start)
        page_label = doc.document[chunk_page].get_label()
        try:
            breadcrumb = doc.get_breadcrumb(chunk_page)
        except IndexError:
            logging.info(f"could not find breadcrumb for page_label {page_label}")
            breadcrumb = ""
            print(chunk)
        return {
                "breadcrumb": breadcrumb,
                "text": chunk.strip(),
                "page_label": page_label,
                "bbox": bbox
            }

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_chunk, start_ix, end_ix) for start_ix, end_ix in chunk_spans]
        for future in concurrent.futures.as_completed(futures):
            yield future.result()



# def main(doc_path, tokenizer_id='mistralai/Mistral-7B-v0.1', max_chunk_len=256, start_page=0, end_page=None):
#     chunker = SentencePreservingChunker(AutoTokenizer.from_pretrained(tokenizer_id), max_chunk_len)
#     document = Document(doc_path)
#     chunks_with_metadata = list(get_document_chunks(document, chunker, start_end=(start_page, end_page)))
#     output_path = f"{doc_path}.chunks.json"

#     with open(output_path, 'w') as f:
#         json.dump(chunks_with_metadata, f, indent=2)

#     return chunks_with_metadata



if __name__ == "__main__":
    fire.Fire(main)
