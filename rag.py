
"""
Lightweight in-memory RAG for per-client hospital FAQ knowledge bases.

Each client's knowledge base is a single text file (see
client_config.csv's `knowledge_base_file` column, exposed as
state["templates"]["_knowledge_base_file"]). At first use, the file is
split into overlapping chunks and each chunk is embedded once via
OpenAI's embeddings API; the (chunk_text, embedding) pairs are cached in
memory, keyed by the file's path + mtime - so an edited file is picked
up automatically without a restart, but an unchanged file is never
re-embedded on every turn (which would be slow and wasteful).

Deliberately dependency-light: no vector database, no numpy - each
client's knowledge base is a single small-to-medium document, not a
large corpus, so a linear cosine-similarity scan in pure Python is
entirely fast enough at this scale. If a clinic's knowledge base ever
grows to genuinely large size (many documents, thousands of chunks),
this module would need to move to a real vector store instead - not a
concern at the current scale.

GENERIC BY DESIGN: nothing here is specific to any one clinic - the
file path is the only per-client input, exactly like doctors_base_url.
Adding a new clinic's knowledge base is just adding a new text file and
pointing client_config.csv's knowledge_base_file column at it.
"""

import logging
import math
import os
import re
from typing import Optional

from langchain_openai import OpenAIEmbeddings

logger = logging.getLogger("rag")

_EMBEDDING_MODEL_NAME = "text-embedding-3-small"
_embeddings_model: Optional[OpenAIEmbeddings] = None

CHUNK_SIZE_CHARS = 800
CHUNK_OVERLAP_CHARS = 150
# Raised from 4: some knowledge-base sections genuinely span many
# chunks - e.g. this clinic's own Ultrasound prep-instructions section
# is ~4,800 chars, roughly 7 chunks at this chunk size - and a caller
# asking a general question about that one exam legitimately needs
# every category's chunk back in the same call, not just whichever
# handful scored highest. CONFIRMED REAL PRODUCTION FAILURE: asked a
# general "تعليمات الموجات فوق الصوتية" question, only the
# echocardiogram sub-category's chunk(s) came back (whatever scored
# highest against a generic query) - abdominal, breast, pelvic,
# pregnancy, prostate, renal, and Doppler were silently absent from the
# passages entirely, so the reply correctly reported only what it was
# given, but what it was given covered a small fraction of a 9-category
# real section. RELEVANCE_FLOOR (below) still filters every candidate
# before it is returned, so raising this does not let through anything
# that scored too low to be relevant - it only lets more genuinely
# relevant chunks through when a section is this large.
DEFAULT_TOP_K = 8

# Cache: file_path -> (mtime, [(chunk_text, embedding_vector), ...])
_CACHE: dict = {}

# Cache for search over a LIVE, already-fetched list of items (e.g. the
# lab/imaging service catalogue) rather than a static file: content_key
# -> [(item_dict, embedding_vector), ...]. There is no mtime to key on
# here since the data isn't a file, so the key is derived from the
# items' own content (see _items_cache_key) - an unchanged catalogue is
# never re-embedded, and a changed one (a service published/unpublished)
# gets a new key and is embedded fresh.
_ITEMS_CACHE: dict = {}
_ITEMS_CACHE_MAX_ENTRIES = 200


def _get_embeddings_model() -> OpenAIEmbeddings:
    """Lazily construct the embeddings client - avoids requiring
    OPENAI_API_KEY at import time (e.g. for tests that never touch RAG)."""

    global _embeddings_model
    if _embeddings_model is None:
        _embeddings_model = OpenAIEmbeddings(model=_EMBEDDING_MODEL_NAME)
    return _embeddings_model


# HEADING-AWARE CHUNKING.
#
# CONFIRMED REAL PRODUCTION FAILURE: asked "مين هما عز لاب" ("who is
# Ezz Lab?"), the best chunk scored 0.306 - just under RELEVANCE_FLOOR -
# and the assistant told the patient it had no information about the
# lab, although the knowledge base has a whole "About Ezz Labs / عن معامل
# عز" section and a "لماذا عز لاب" block. Cause: the old chunker packed
# blank-line paragraphs greedily with no notion of headings, so a
# heading routinely landed as the LAST line of the previous chunk and
# the body under it was embedded without it - no chunk put the lab's
# name next to the text describing it. It also emitted heading-only
# fragments (96 chars of "## Home Page / ### English / **Why Ezz lab**")
# that carry no answer at all.
#
# Now:
#   1. Every markdown heading (#..######) is a hard chunk boundary -
#      one chunk never spans two sections.
#   2. Every chunk starts with its heading path, e.g.
#      "About Ezz Labs / عن معامل عز › العربية / Arabic", so a passage
#      carries its own context both for embedding and for the model
#      that reads it.
#   3. A short standalone **bold** line is a sub-heading: it is never
#      left as the last line of a chunk - it moves to the next chunk,
#      beside the text it introduces.
#   4. A heading with no body of its own produces no chunk; it lives on
#      in the path of the chunks below it.
#
# Still generic: this only reads markdown structure, never a clinic's
# wording. RAG_HEADING_AWARE_CHUNKS=0 restores the old chunker without
# a redeploy (restart needed - embeddings are cached in memory).
HEADING_AWARE_CHUNKS = os.getenv("RAG_HEADING_AWARE_CHUNKS", "1").strip().lower() not in ("0", "false", "no", "off")

_MD_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
_BOLD_LINE_RE = re.compile(r"^\*\*([^*]+)\*\*:?$")
_PATH_SEPARATOR = " › "
# The document title (a single "# ..." line) names the whole file, so
# it adds nothing to distinguish one chunk from another.
_PATH_MIN_LEVEL = 2


def _is_bold_subheading(line: str) -> bool:
    match = _BOLD_LINE_RE.match(line)
    if not match:
        return False
    inner = match.group(1).strip()
    # A bold phone number or hotline ("**15032**") is content, not a
    # heading - it must stay with whatever it belongs to.
    return 0 < len(inner) <= 80 and bool(re.search(r"[^\W\d_]", inner))


def _split_into_sections(text: str) -> list:
    """[(heading_path, [block, ...]), ...] in document order. A block is
    a blank-line paragraph or a standalone bold sub-heading line."""

    path: dict = {}
    sections = []
    blocks: list = []
    paragraph: list = []

    def end_paragraph():
        if paragraph:
            blocks.append("\n".join(paragraph))
            paragraph.clear()

    def end_section():
        end_paragraph()
        if blocks:
            heading_path = _PATH_SEPARATOR.join(path[level] for level in sorted(path) if level >= _PATH_MIN_LEVEL)
            sections.append((heading_path, blocks.copy()))
            blocks.clear()

    for raw_line in text.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        line = raw_line.strip()

        heading = _MD_HEADING_RE.match(line)
        if heading:
            end_section()
            level = len(heading.group(1))
            for deeper in [lvl for lvl in path if lvl >= level]:
                del path[deeper]
            path[level] = heading.group(2).strip()
            continue

        if not line:
            end_paragraph()
            continue

        if _is_bold_subheading(line):
            end_paragraph()
            blocks.append(line)
            continue

        paragraph.append(line)

    end_section()
    return sections


def _chunk_text(text: str) -> list:
    """Split text into chunks that respect the document's headings (see
    HEADING-AWARE CHUNKING above). Within a section, paragraphs are
    packed up to CHUNK_SIZE_CHARS; a single paragraph longer than that
    is hard-split with CHUNK_OVERLAP_CHARS overlap, as before."""

    if not HEADING_AWARE_CHUNKS:
        return _chunk_text_legacy(text)

    chunks = []

    for heading_path, blocks in _split_into_sections(text):
        prefix = (heading_path + "\n") if heading_path else ""
        budget = max(CHUNK_SIZE_CHARS - len(prefix), 200)
        current: list = []

        def flush():
            carried = []
            # Never end a chunk on a sub-heading - it belongs with the
            # text after it.
            while current and _is_bold_subheading(current[-1]):
                carried.insert(0, current.pop())
            if current:
                chunks.append(prefix + "\n".join(current))
            current.clear()
            current.extend(carried)

        for block in blocks:
            if len(block) > budget:
                flush()
                lead = "\n".join(current)
                current.clear()
                step = budget - CHUNK_OVERLAP_CHARS
                body = (lead + "\n" + block) if lead else block
                for i in range(0, len(body), step):
                    chunks.append(prefix + body[i:i + budget])
                    if i + budget >= len(body):
                        break
                continue

            size = sum(len(b) + 1 for b in current) + len(block)
            if current and size > budget:
                flush()
            current.append(block)

        flush()
        # A section that ends on bold lines with nothing after them. A
        # run of several is real content (e.g. a list of partner
        # names) - keep it. A single one is a heading with no text
        # under it ("**Quality Assurance**"): it answers nothing and
        # would only take a top_k slot from a real passage.
        if len(current) > 1:
            chunks.append(prefix + "\n".join(current))

    return chunks


def _chunk_text_legacy(text: str) -> list:
    """Split text into chunks along paragraph/blank-line boundaries
    where possible (keeps related sentences together), falling back to
    a hard character-count split with overlap for any single paragraph
    longer than the chunk size on its own."""

    paragraphs = re.split(r"\n\s*\n", text)
    chunks = []
    current = ""

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        if len(current) + len(para) + 1 <= CHUNK_SIZE_CHARS:
            current = (current + "\n" + para).strip()
            continue

        if current:
            chunks.append(current)
            current = ""

        if len(para) <= CHUNK_SIZE_CHARS:
            current = para
        else:
            step = CHUNK_SIZE_CHARS - CHUNK_OVERLAP_CHARS
            for i in range(0, len(para), step):
                chunks.append(para[i:i + CHUNK_SIZE_CHARS])

    if current:
        chunks.append(current)

    return chunks


def _load_and_embed(file_path: str) -> list:
    with open(file_path, encoding="utf-8") as f:
        text = f.read()

    chunks = _chunk_text(text)
    if not chunks:
        return []

    logger.info("rag: embedding %d chunk(s) for %s", len(chunks), file_path)

    try:
        vectors = _get_embeddings_model().embed_documents(chunks)
    except Exception:
        logger.exception("rag: failed to embed knowledge base chunks for %s", file_path)
        # Signalled as None, NOT as []. The two mean different things
        # and the caller must be able to tell them apart - see
        # _get_cached_chunks.
        return None

    return list(zip(chunks, vectors))


def _get_cached_chunks(file_path: str) -> list:
    try:
        mtime = os.path.getmtime(file_path)
    except OSError:
        logger.warning("rag: knowledge base file not found: %s", file_path)
        return []

    cached = _CACHE.get(file_path)
    if cached and cached[0] == mtime:
        return cached[1]

    chunks_with_vectors = _load_and_embed(file_path)

    # ONLY CACHE A SUCCESS.
    #
    # A transient embeddings failure (a rate limit, a blip) used to be
    # cached as an empty corpus against the file's own mtime - and since
    # the file's mtime does not change, the knowledge base stayed empty
    # for the rest of the process's life. Every FAQ question after that
    # returned "not_found", so the assistant told patients this clinic
    # has no information on subjects its knowledge base documents in
    # detail. A false statement about the clinic, produced by a cache.
    if chunks_with_vectors is None:
        return []

    _CACHE[file_path] = (mtime, chunks_with_vectors)
    return chunks_with_vectors


def _cosine_similarity(a: list, b: list) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def list_services(file_path: str, language: str = "ar") -> list:
    """Return the knowledge base's own list of top-level SERVICES, in
    document order.

    WHY THIS EXISTS: "what services do you offer?" is a
    table-of-contents question, and semantic search is the wrong tool
    for it. `search()` returns the passages most similar to the query,
    which for this question means a handful of DETAIL paragraphs -
    confirmed in production, the answer came back as a bulleted mix of
    inpatient amenities (garden, gym, art therapy area, isolation
    rooms) while four of the clinic's six actual services were never
    mentioned at all. Amenities are not services, and a partial list
    misrepresents what the clinic offers.

    The knowledge base is already structured for this: a numbered
    top-level section whose heading names services ("4. الخدمات |
    Services"), with each service as a numbered subheading beneath it
    ("4.1 خدمة الطوارئ النفسية | Psychiatric Emergency Service"). This
    reads those subheadings directly, so the list is always complete,
    always in the clinic's own wording, and always in the clinic's own
    order.

    Returns [] when the file is missing or has no recognizable services
    section - the caller then falls back to normal semantic search.
    """

    if not file_path:
        return []

    try:
        with open(file_path, encoding="utf-8") as f:
            text = f.read()
    except OSError:
        logger.warning("rag: knowledge base file not found for service listing: %s", file_path)
        return []

    lines = [line.strip() for line in text.replace("\r\n", "\n").replace("\r", "\n").split("\n")]

    section_number = None
    for line in lines:
        match = re.match(r"^(\d+)\.\s+(.*)$", line)
        if not match:
            continue
        heading = match.group(2)
        # Heading must NAME services - not merely mention the word in a
        # sentence, so keep this to short heading-like lines.
        if len(heading) <= 80 and re.search(r"الخدمات|services", heading, re.IGNORECASE):
            section_number = match.group(1)
            break

    if section_number is None:
        logger.info("rag: no services section heading found in %s", file_path)
        return []

    services = []
    for line in lines:
        match = re.match(rf"^{section_number}\.(\d+)\s+(.+)$", line)
        if not match:
            continue

        heading = match.group(2).strip()
        arabic_part, _, english_part = heading.partition("|")
        arabic_part = arabic_part.strip()
        english_part = english_part.strip()

        name = (english_part or arabic_part) if language == "en" else (arabic_part or english_part)
        if name:
            services.append(name)

    logger.info("rag: %d service(s) read from the services section of %s", len(services), file_path)

    return services


# THE RELEVANCE FLOOR.
#
# `search()` used to return the top_k chunks unconditionally, discarding
# the score - so "the four least-dissimilar paragraphs in the document"
# was reported to the caller as "found", for ANY question, including
# ones the knowledge base says nothing about. The caller
# (tools.answer_hospital_faq) then handed those passages to the model
# with an instruction to summarize them naturally, and the model - with
# no signal that they did not answer the question - wrote a fluent,
# confident answer out of unrelated text. That is the textbook RAG
# failure, and it made the tool's own "not_found" branch unreachable:
# the only way to get it was a missing file.
#
# Calibrated for text-embedding-3-small cosine similarity, where
# genuinely unrelated short-query/passage pairs sit well below 0.3.
# Tunable without a redeploy because the right floor depends on how a
# given clinic writes its knowledge base.
RELEVANCE_FLOOR = float(os.getenv("RAG_RELEVANCE_FLOOR", "0.32"))


def _items_cache_key(items: list) -> str:
    """A content-derived cache key for a live list of {"id", "name",
    "description"} dicts - two calls with the same real catalogue
    content (regardless of order the API happened to return it in this
    time) hit the same cache entry."""

    basis = "|".join(
        sorted(
            f"{i.get('id')}:{i.get('name')}:{i.get('description') or ''}"
            for i in items
        )
    )
    return str(hash(basis))


def search_items(items: list, query: str, min_score: float = RELEVANCE_FLOOR) -> list:
    """Semantic search over a REAL, already-fetched list of item dicts
    (each needs at least "name"; "id"/"description" are used if
    present) - e.g. a live lab/imaging service catalogue from
    api.get_services. Never invents an item that isn't in `items`.

    Unlike `search()`/`search_with_scores()` (capped at `top_k`), this
    returns EVERY item at or above `min_score`, most relevant first -
    by design, so a broad query like "تحليل دم" surfaces every real
    matching test (CBC, WBC, RBC, ...) rather than an arbitrary top few.

    Returns [(item, score), ...]. Embeddings for a given catalogue's
    content are cached in memory (see _items_cache_key) so repeated
    searches against an unchanged catalogue don't re-embed it.
    """

    if not items:
        return []

    key = _items_cache_key(items)
    cached = _ITEMS_CACHE.get(key)

    if cached is None:
        texts = [
            f"{i.get('name') or ''}. {i.get('description') or ''}".strip()
            for i in items
        ]
        try:
            vectors = _get_embeddings_model().embed_documents(texts)
        except Exception:
            logger.exception(
                "rag: failed to embed live item catalogue (%d item(s))", len(items),
            )
            return []

        cached = list(zip(items, vectors))

        # Bound the cache - a live catalogue can change over the life of
        # a long-running process (services published/unpublished), so
        # old content-keys would otherwise accumulate forever. Evicting
        # the oldest entry once the bound is hit is enough; this is a
        # small in-memory speed cache, not a source of truth.
        if len(_ITEMS_CACHE) >= _ITEMS_CACHE_MAX_ENTRIES:
            _ITEMS_CACHE.pop(next(iter(_ITEMS_CACHE)))
        _ITEMS_CACHE[key] = cached

    try:
        query_vector = _get_embeddings_model().embed_query(query)
    except Exception:
        logger.exception("rag: failed to embed item search query %r", query)
        return []

    scored = [
        (item, _cosine_similarity(query_vector, vector))
        for item, vector in cached
    ]
    scored.sort(key=lambda pair: pair[1], reverse=True)

    relevant = [pair for pair in scored if pair[1] >= min_score]

    if not relevant:
        logger.info(
            "rag: item search - nothing cleared the relevance floor (%.2f) for "
            "query %r among %d item(s) - best score was %.3f",
            min_score, query[:120], len(items), scored[0][1] if scored else 0.0,
        )

    return relevant


def search(file_path: str, query: str, top_k: int = DEFAULT_TOP_K) -> list:
    """Return the RELEVANT chunks for `query`, most relevant first.

    Only passages scoring at or above `RELEVANCE_FLOOR` are returned, so
    an empty list genuinely means "this knowledge base does not answer
    that" rather than "here are the four closest paragraphs regardless".

    Returns [] if the file is missing/empty, if nothing clears the floor,
    or if the query/chunks couldn't be embedded (a transient API error -
    see `search_with_scores` for callers that need to tell those apart).
    """

    return [chunk for chunk, _score in search_with_scores(file_path, query, top_k)]


def search_with_scores(file_path: str, query: str, top_k: int = DEFAULT_TOP_K) -> list:
    """`search()`, but returns [(chunk, score), ...] so a caller can
    report HOW well the knowledge base matched - the difference between
    "we have nothing on that" and "here is the answer"."""

    if not file_path:
        return []

    chunks_with_vectors = _get_cached_chunks(file_path)
    if not chunks_with_vectors:
        return []

    try:
        query_vector = _get_embeddings_model().embed_query(query)
    except Exception:
        logger.exception("rag: failed to embed query %r", query)
        return []

    scored = [
        (chunk, _cosine_similarity(query_vector, vector))
        for chunk, vector in chunks_with_vectors
    ]
    scored.sort(key=lambda pair: pair[1], reverse=True)

    relevant = [pair for pair in scored[:top_k] if pair[1] >= RELEVANCE_FLOOR]

    if not relevant:
        logger.info(
            "rag: nothing cleared the relevance floor (%.2f) for query %r - "
            "best score was %.3f. Reporting no match rather than returning "
            "the closest unrelated passages.",
            RELEVANCE_FLOOR, query[:120], scored[0][1] if scored else 0.0,
        )

    return relevant
