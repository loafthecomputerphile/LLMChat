from bs4 import BeautifulSoup, Tag
import re


READABLE_ATTRS: tuple[str, str, str, str] = ("title", "alt", "aria-label", "placeholder")

REMOVE_TAGS: set[str] = {
    "script", "style", "noscript", "iframe",
    "svg", "canvas", "template",
    "header", "footer", "nav", "aside", "link"
}

BLOCK_TAGS: set[str] = {
    "article", "main", "section", "p", "pre", "code",
    "div", "li", "ul", "ol", "table", "tr", "td", "th",
    "blockquote", "h1", "h2", "h3", "h4", "h5", "h6",
    "label", "button", "a"
}


WHITESPACE_RE: re.Pattern = re.compile(r"\s+")
def normalize(text: str) -> str:
    return WHITESPACE_RE.sub(" ", text).strip()

def prune_by_attr(soup: BeautifulSoup, rules: list[tuple[str, str]]) -> None:
    
    for attr, value in rules:
        for tag in filter(lambda tag: tag.attrs is not None, soup.find_all(attrs={attr: True})):
            attr_val = tag.get(attr)

            if isinstance(attr_val, (list, tuple, set)):
                if value in attr_val: tag.decompose()
                continue
            
            if str(attr_val) == value:
                tag.decompose()



def extract_readable_text(html: str, prune_rules: list[tuple[str, str]] | None = None) -> str:
    if not html:
        return ""

    soup: BeautifulSoup = BeautifulSoup(html, "lxml")

    # Remove clearly non-readable elements
    for tag in soup.find_all(REMOVE_TAGS):
        tag.decompose()
        
    if prune_rules:
        prune_by_attr(soup, prune_rules)

    collected: list[str] = []

    def collect_from_tag(tag: Tag) -> list[str]:
        parts: list[str] = []

        # Main visible text
        text: str = normalize(tag.get_text(" ", strip=True))
        if text:
            parts.append(text)

        # User-visible attributes (critical for <a>, <img>, inputs)
        for attr in filter(tag.has_attr, READABLE_ATTRS):
            attr_text: str = normalize(str(tag[attr]))
            if attr_text and attr_text not in text:
                parts.append(attr_text)

        return parts

    # Traverse in document order
    for tag in soup.find_all(True):
        if tag.name in REMOVE_TAGS:
            continue

        # Skip empty / invisible tags
        if not tag.get_text(strip=True) and not any(tag.has_attr(a) for a in READABLE_ATTRS):
            continue

        # Prefer semantic blocks but allow inline containers (like <a>)
        if tag.name in BLOCK_TAGS or tag.name == "a":
            for p in collect_from_tag(tag):
                collected.append(p)

    # Fallback: raw visible text if extraction failed
    if not collected:
        text: str = normalize(soup.get_text("\n", strip=True))
        collected: list[str] = [l for l in text.splitlines() if len(l) >= 16]

    # Deduplicate while preserving order
    seen: set[str] = set()
    result: list[str] = []
    for line in filter(lambda x: x not in seen, collected):
        seen.add(line)
        result.append(line)

    return "\n\n".join(result)



wiki_prunes: list[tuple[str, str]] = [
    ("class", "reference"),
    ("class", "references"),
    ("role", "navigation"),
]
