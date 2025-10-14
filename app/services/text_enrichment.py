"""
Text enrichment service for Charlie.

Processes episode content through markdown rendering and entity mention highlighting.
Transforms raw markdown content into semantically enriched HTML with inline entity highlights.
"""

import logging
import re
from typing import List, Optional
from html.parser import HTMLParser
from io import StringIO

import markdown
from markdown.extensions import fenced_code, tables, nl2br

from app.models.graph import EntityNode
from app.utils.mention_finder import (
    find_entity_mentions,
    EntityMention,
)

logger = logging.getLogger(__name__)


class TextExtractor(HTMLParser):
    """
    HTML parser that extracts text content to match positions used by mention finder.

    This ensures text extraction is deterministic and matches the structure
    that HTMLTextInjector will operate on.
    """

    def __init__(self):
        super().__init__()
        self.text_parts = []
        self.in_code = False
        self.in_pre = False

    def handle_starttag(self, tag, attrs):
        """Track if we're entering code/pre blocks."""
        if tag in ("code", "pre"):
            if tag == "code":
                self.in_code = True
            if tag == "pre":
                self.in_pre = True

    def handle_endtag(self, tag):
        """Track if we're leaving code/pre blocks."""
        if tag == "code":
            self.in_code = False
        if tag == "pre":
            self.in_pre = False

    def handle_data(self, data):
        """Collect text data from all nodes."""
        self.text_parts.append(data)

    def get_text(self):
        """Get the extracted text content."""
        return "".join(self.text_parts)


class HTMLTextInjector(HTMLParser):
    """
    HTML parser that injects entity mention spans into text nodes.

    This preserves HTML structure while adding entity highlights only in text content,
    not in tags, attributes, or code blocks.
    """

    def __init__(self, mentions: List[EntityMention]):
        super().__init__()
        self.mentions = mentions
        self.output = StringIO()
        self.in_code = False
        self.in_pre = False
        self.current_position = 0
        self.text_offset = 0  # Track position in text-only content

    def handle_starttag(self, tag, attrs):
        """Handle opening tags."""
        self.output.write(self.get_starttag_text())

        # Track if we're in code/pre blocks (skip entity injection here)
        if tag in ("code", "pre"):
            if tag == "code":
                self.in_code = True
            if tag == "pre":
                self.in_pre = True

    def handle_endtag(self, tag):
        """Handle closing tags."""
        self.output.write(f"</{tag}>")

        if tag == "code":
            self.in_code = False
        if tag == "pre":
            self.in_pre = False

    def handle_data(self, data):
        """Handle text content - inject entity mentions here."""
        # Skip injection in code blocks
        if self.in_code or self.in_pre:
            self.output.write(data)
            return

        # Find mentions that fall within this text node
        text_start = self.text_offset
        text_end = self.text_offset + len(data)

        # Get mentions relevant to this text segment
        relevant_mentions = [
            m for m in self.mentions if m.start >= text_start and m.end <= text_end
        ]

        if not relevant_mentions:
            # No mentions, write data as-is
            self.output.write(data)
        else:
            # Inject mentions into this text node
            result = data
            offset = 0

            for mention in relevant_mentions:
                # Calculate local position within this text node
                local_start = mention.start - text_start + offset
                local_end = mention.end - text_start + offset

                # Build entity span
                span_open = (
                    f'<span class="entity-mention" '
                    f'data-entity-uuid="{mention.entity_uuid}" '
                    f'data-entity-name="{_escape_html(mention.entity_name)}" '
                    f'data-entity-summary="{_escape_html(mention.entity_summary)}">'
                )
                span_close = "</span>"

                # Inject span
                result = (
                    result[:local_start]
                    + span_open
                    + result[local_start:local_end]
                    + span_close
                    + result[local_end:]
                )

                # Adjust offset for next mention
                offset += len(span_open) + len(span_close)

            self.output.write(result)

        # Update text position tracker
        self.text_offset += len(data)

    def handle_startendtag(self, tag, attrs):
        """Handle self-closing tags."""
        self.output.write(self.get_starttag_text())

    def get_result(self):
        """Get the final HTML output."""
        return self.output.getvalue()


class TextEnrichmentService:
    """
    Service for processing episode content with markdown and entity enrichment.
    """

    def __init__(self):
        """Initialize the markdown processor with extensions."""
        self.md = markdown.Markdown(
            extensions=[
                "fenced_code",
                "tables",
                "nl2br",
                "sane_lists",
            ],
            output_format="html5",
        )

    def enrich_episode_content(
        self,
        content: str,
        entities: List[EntityNode],
        enable_entity_highlighting: bool = True,
    ) -> str:
        """
        Process episode content through markdown and entity enrichment pipeline.

        Pipeline:
        1. Convert markdown to HTML
        2. Extract plain text from HTML
        3. Find entity mentions in extracted text
        4. Inject entity spans into HTML text nodes
        5. Return enriched HTML

        Args:
            content: Raw markdown content from episode
            entities: List of entities mentioned in this episode
            enable_entity_highlighting: Whether to highlight entity mentions

        Returns:
            Enriched HTML ready for display
        """
        try:
            if not content:
                return ""

            # Step 1: Convert markdown to HTML
            html = self.md.reset().convert(content)

            # Step 2 & 3: Find entity mentions in HTML text (if enabled)
            mentions = []
            if enable_entity_highlighting and entities:
                # Extract text from HTML to get correct positions
                text_content = self._extract_text_from_html(html)

                # Find entity mentions in that extracted text
                mentions = find_entity_mentions(text_content, entities)
                logger.debug(
                    f"Found {len(mentions)} entity mentions in content "
                    f"({len(entities)} entities provided)"
                )

            # Step 4: Inject entity highlights (if any mentions found)
            if mentions:
                html = self._inject_mentions_in_html(html, mentions)

            return html

        except Exception as e:
            logger.error(f"Error enriching content: {e}", exc_info=True)
            # Fallback to basic markdown without entity highlighting
            return self.md.reset().convert(content)

    def _inject_mentions_in_html(self, html: str, mentions: List[EntityMention]) -> str:
        """
        Inject entity mention spans into rendered HTML.

        This uses HTMLParser to safely inject spans only in text nodes,
        preserving the HTML structure and avoiding breaking tags or attributes.

        Args:
            html: Rendered HTML from markdown
            mentions: List of entity mentions with positions matching HTML text

        Returns:
            HTML with entity spans injected
        """
        try:
            # Use custom HTML parser to inject mentions at their positions
            injector = HTMLTextInjector(mentions)
            injector.feed(html)

            return injector.get_result()

        except Exception as e:
            logger.error(f"Error injecting mentions: {e}", exc_info=True)
            # Fallback: return HTML without entity highlighting
            return html

    def _extract_text_from_html(self, html: str) -> str:
        """
        Extract plain text content from HTML for position mapping.

        Uses the same HTMLParser structure as HTMLTextInjector to ensure
        text positions are consistent between extraction and injection.

        Args:
            html: HTML content

        Returns:
            Plain text content
        """
        extractor = TextExtractor()
        extractor.feed(html)
        return extractor.get_text()

    def generate_preview(self, content: str, max_chars: int = 200) -> str:
        """
        Generate a plain-text preview of markdown content.

        Strips markdown formatting and returns first N characters.

        Args:
            content: Raw markdown content
            max_chars: Maximum characters to include

        Returns:
            Preview text with ellipsis if truncated
        """
        # Remove markdown headers (entire lines)
        text = re.sub(r"^#+\s+.+$", "", content, flags=re.MULTILINE)

        # Remove markdown links
        text = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", text)

        # Remove markdown emphasis
        text = re.sub(r"[*_]{1,2}([^*_]+)[*_]{1,2}", r"\1", text)

        # Remove code blocks
        text = re.sub(r"```[^`]+```", "", text, flags=re.DOTALL)
        text = re.sub(r"`([^`]+)`", r"\1", text)

        # Collapse whitespace
        text = " ".join(text.split())

        # Truncate
        if len(text) > max_chars:
            # Find last space before max_chars
            truncate_at = text.rfind(" ", 0, max_chars)
            if truncate_at > max_chars - 20:  # Only use space if it's close
                text = text[:truncate_at] + "..."
            else:
                text = text[:max_chars] + "..."

        return text.strip()


def _escape_html(text: str) -> str:
    """Escape HTML special characters for safe attribute values."""
    if not text:
        return ""
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&#39;")
    )


# Singleton instance
_enrichment_service: Optional[TextEnrichmentService] = None


def get_enrichment_service() -> TextEnrichmentService:
    """Get or create the text enrichment service singleton."""
    global _enrichment_service
    if _enrichment_service is None:
        _enrichment_service = TextEnrichmentService()
    return _enrichment_service
