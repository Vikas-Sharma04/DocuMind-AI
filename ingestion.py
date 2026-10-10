import json
import logging
import os
import re
from html.parser import HTMLParser
from typing import List, Optional
from urllib.parse import urlparse

import requests
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

MAX_WEB_PAGE_BYTES = 5 * 1024 * 1024
WEB_REQUEST_TIMEOUT_SECONDS = 15


class _WebPageTextParser(HTMLParser):
    """Extract readable text and a page title from HTML."""

    _ignored_tags = {"script", "style", "noscript", "svg"}
    _block_tags = {
        "article", "blockquote", "br", "div", "h1", "h2", "h3", "h4",
        "li", "main", "p", "section", "td", "th", "tr",
    }

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.text_parts: List[str] = []
        self.title_parts: List[str] = []
        self.structured_scripts: List[str] = []
        self._script_parts: List[str] = []
        self._capture_script = False
        self._ignored_stack: List[str] = []
        self._in_title = False

    def handle_starttag(self, tag: str, attrs) -> None:
        if self._ignored_stack:
            if tag in self._ignored_tags:
                self._ignored_stack.append(tag)
            return
        if tag in self._ignored_tags:
            if tag == "script":
                attributes = dict(attrs)
                script_type = (attributes.get("type") or "").lower()
                self._capture_script = (
                    attributes.get("id") == "__NEXT_DATA__"
                    or script_type in {"application/json", "application/ld+json"}
                )
                self._script_parts = []
            self._ignored_stack.append(tag)
        elif tag == "title":
            self._in_title = True
        elif tag in self._block_tags:
            self.text_parts.append(" ")

    def handle_endtag(self, tag: str) -> None:
        if self._ignored_stack:
            if tag in self._ignored_stack:
                index = len(self._ignored_stack) - 1 - self._ignored_stack[::-1].index(tag)
                del self._ignored_stack[index:]
                if tag == "script" and self._capture_script:
                    self.structured_scripts.append("".join(self._script_parts))
                    self._capture_script = False
                    self._script_parts = []
            return
        if tag == "title":
            self._in_title = False
        elif tag in self._block_tags:
            self.text_parts.append(" ")

    def handle_data(self, data: str) -> None:
        if self._ignored_stack:
            if self._capture_script:
                self._script_parts.append(data)
            return
        if self._in_title:
            self.title_parts.append(data)
        self.text_parts.append(data)


def load_web_page(url: str) -> List[Document]:
    """Fetch one HTTP(S) page and return its readable content as a document."""
    parsed_url = urlparse(url.strip())
    if parsed_url.scheme not in {"http", "https"} or not parsed_url.hostname:
        raise ValueError("Enter a valid webpage URL beginning with http:// or https://.")
    if parsed_url.username or parsed_url.password:
        raise ValueError("Webpage URLs must not contain embedded credentials.")

    try:
        with requests.get(
            url.strip(),
            headers={"User-Agent": "Mozilla/5.0 (compatible; DocuMindAI/1.0)"},
            timeout=WEB_REQUEST_TIMEOUT_SECONDS,
            stream=True,
        ) as response:
            response.raise_for_status()
            content_type = response.headers.get("Content-Type", "").lower()
            if content_type and "html" not in content_type:
                raise ValueError(
                    f"The URL did not return an HTML page (content type: {content_type})."
                )

            page_bytes = bytearray()
            for data in response.iter_content(chunk_size=64 * 1024):
                page_bytes.extend(data)
                if len(page_bytes) > MAX_WEB_PAGE_BYTES:
                    raise ValueError("The webpage is larger than the 5 MB limit.")

            encoding = response.encoding or "utf-8"
            if "charset=" not in content_type:
                encoding = response.apparent_encoding or encoding
            html = bytes(page_bytes).decode(encoding, errors="replace")
            final_url = response.url
    except requests.RequestException as exc:
        logger.error("Webpage loading error for %s: %s", url, exc)
        status_code = getattr(exc.response, "status_code", None)
        if status_code in {401, 403}:
            raise ValueError(
                f"The website denied access (HTTP {status_code}). It may block "
                "automated requests or require sign-in. Webpage loaders, including "
                "WebBaseLoader, use server-side requests and cannot read a page "
                "that the website denies. Try a publicly accessible page."
            ) from exc
        raise ValueError(f"Failed to fetch webpage: {exc}") from exc

    parser = _WebPageTextParser()
    parser.feed(html)
    title = re.sub(r"\s+", " ", " ".join(parser.title_parts)).strip()
    page_text = re.sub(r"\s+", " ", " ".join(parser.text_parts)).strip()
    structured_text = []

    def add_json_strings(value) -> None:
        if isinstance(value, dict):
            for nested_value in value.values():
                add_json_strings(nested_value)
        elif isinstance(value, list):
            for nested_value in value:
                add_json_strings(nested_value)
        elif isinstance(value, str) and len(value.strip()) > 20:
            candidate = value.strip()
            if "<" in candidate and ">" in candidate:
                nested_parser = _WebPageTextParser()
                nested_parser.feed(candidate)
                candidate = " ".join(nested_parser.text_parts)
            candidate = re.sub(r"\s+", " ", candidate).strip()
            if candidate:
                structured_text.append(candidate)

    for script in parser.structured_scripts:
        try:
            add_json_strings(json.loads(script))
        except json.JSONDecodeError:
            logger.debug("Skipping non-JSON structured webpage script")
    if structured_text:
        page_text = re.sub(
            r"\s+", " ", f"{page_text} {' '.join(structured_text)}"
        ).strip()
    if not page_text:
        raise ValueError("The webpage contains no readable text.")

    content = f"Title: {title}\n\n{page_text}" if title else page_text
    logger.info("Loaded webpage '%s' from %s", title or final_url, final_url)
    return [
        Document(
            page_content=content,
            metadata={"source": final_url, "title": title, "page": 1},
        )
    ]


def load_pdf(
    file_path: str,
    max_pages: Optional[int] = None
) -> List[Document]:
    """
    Load a PDF file and extract documents

    Args:
        file_path: Path to the PDF file
        max_pages: Optional limit on number of pages to load

    Returns:
        List of Document objects with non-empty content
    """
    try:
        if not os.path.exists(file_path):
            raise ValueError(f"File not found: {file_path}")

        logger.info(f"Loading PDF: {file_path}")

        loader = PyPDFLoader(file_path)

        docs = []
        for i, doc in enumerate(loader.lazy_load()):
            if max_pages and i >= max_pages:
                logger.info(f"Stopped loading at {max_pages} pages")
                break

            if doc.page_content and doc.page_content.strip():
                docs.append(doc)

        if not docs:
            raise ValueError("PDF contains no readable text")

        logger.info(f"Loaded {len(docs)} valid pages")
        return docs

    except Exception as e:
        logger.error(f"PDF loading error: {str(e)}")
        raise Exception(f"Failed to load PDF: {str(e)}")


def validate_pdf(file_path: str) -> bool:
    """
    Lightweight PDF validation

    Args:
        file_path: Path to file

    Returns:
        True if valid PDF, False otherwise
    """
    try:
        if not os.path.exists(file_path):
            return False

        with open(file_path, "rb") as f:
            header = f.read(5)

        return header == b"%PDF-"

    except Exception as e:
        logger.error(f"PDF validation failed: {str(e)}")
        return False