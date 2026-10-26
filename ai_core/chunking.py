
import re


def chunk_text(
    text: str,
    chunk_size: int = 900,
    chunk_overlap: int = 150,
) -> list[str]:
    """Split text into chunks while preserving Markdown section context."""

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")

    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError(
            "chunk_overlap must be non-negative and smaller than chunk_size."
        )

    text = text.strip()
    if not text:
        return []

    sections = re.split(r"(?m)(?=^#{1,6}\s+)", text)
    chunks: list[str] = []

    def split_long_section(section: str) -> list[str]:
        lines = section.strip().splitlines()
        heading = ""
        body_lines = lines

        if lines and re.match(r"^#{1,6}\s+", lines[0]):
            heading = lines[0].strip()
            body_lines = lines[1:]

        body = "\n".join(body_lines).strip()
        prefix = f"{heading}\n\n" if heading else ""
        available_size = chunk_size - len(prefix)

        if available_size <= 0:
            return [section[:chunk_size]]

        if len(prefix) + len(body) <= chunk_size:
            return [section.strip()]

        paragraphs = [
            p.strip()
            for p in re.split(r"\n\s*\n", body)
            if p.strip()
        ]

        pieces: list[str] = []
        buffer = ""

        def emit_buffer() -> None:
            nonlocal buffer
            if buffer.strip():
                pieces.append(prefix + buffer.strip())
                buffer = ""

        for paragraph in paragraphs:
            if len(paragraph) > available_size:
                emit_buffer()
                start = 0

                while start < len(paragraph):
                    end = min(start + available_size, len(paragraph))
                    piece = paragraph[start:end].strip()

                    if piece:
                        pieces.append(prefix + piece)

                    if end == len(paragraph):
                        break

                    start = end - min(
                        chunk_overlap,
                        available_size - 1,
                    )
                continue

            candidate = (
                f"{buffer}\n\n{paragraph}" if buffer else paragraph
            )

            if len(candidate) <= available_size:
                buffer = candidate
            else:
                emit_buffer()
                buffer = paragraph

        emit_buffer()
        return pieces

    for section in sections:
        section = section.strip()
        if not section:
            continue

        if len(section) <= chunk_size:
            chunks.append(section)
        else:
            chunks.extend(split_long_section(section))

    # Merge consecutive chunks from the same section when they fit.
    # Remove duplicated heading prefixes before attempting the merge.
    merged: list[str] = []

    for chunk in chunks:
        if merged:
            previous = merged[-1]

            previous_heading = re.match(r"^(#{1,6}\s+[^\n]+)\n\n", previous)
            current_heading = re.match(r"^(#{1,6}\s+[^\n]+)\n\n", chunk)

            if (
                previous_heading
                and current_heading
                and previous_heading.group(1) == current_heading.group(1)
            ):
                heading = previous_heading.group(1)
                previous_body = previous[len(heading):].strip()
                current_body = chunk[len(heading):].strip()
                candidate = f"{heading}\n\n{previous_body}\n\n{current_body}"

                if len(candidate) <= chunk_size:
                    merged[-1] = candidate
                    continue

        merged.append(chunk)

    return merged
