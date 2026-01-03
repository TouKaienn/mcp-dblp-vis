import contextlib
import difflib
import logging
import re
from collections import Counter
from typing import Any

import requests

logger = logging.getLogger("dblp_client")

# Default timeout for all HTTP requests
REQUEST_TIMEOUT = 10  # seconds

# Headers for DBLP API requests
# DBLP recommends using an identifying User-Agent to avoid rate-limiting
# See: https://dblp.org/faq/1474706.html
HEADERS = {
    "User-Agent": "mcp-dblp/1.1.1 (https://github.com/szeider/mcp-dblp)",
    "Accept": "application/json",
}


def _fetch_publications(single_query: str, max_results: int) -> list[dict[str, Any]]:
    """Helper function to fetch publications for a single query string."""
    results = []
    try:
        url = "https://dblp.org/search/publ/api"
        params = {"q": single_query, "format": "json", "h": max_results}
        response = requests.get(url, params=params, headers=HEADERS, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        data = response.json()
        hits = data.get("result", {}).get("hits", {})
        total = int(hits.get("@total", "0"))
        logger.info(f"Found {total} results for query: {single_query}")
        if total > 0:
            publications = hits.get("hit", [])
            if not isinstance(publications, list):
                publications = [publications]
            for pub in publications:
                info = pub.get("info", {})
                authors = []
                authors_data = info.get("authors", {}).get("author", [])
                if not isinstance(authors_data, list):
                    authors_data = [authors_data]
                for author in authors_data:
                    if isinstance(author, dict):
                        authors.append(author.get("text", ""))
                    else:
                        authors.append(str(author))

                # Extract the proper DBLP URL or ID for BibTeX retrieval
                dblp_url = info.get("url", "")
                dblp_key = ""

                if dblp_url:
                    # Extract the key from the URL (e.g., https://dblp.org/rec/journals/jmlr/ChowdheryNDBMGMBCDDRSSTWPLLNSZDYJGKPSN23)
                    dblp_key = dblp_url.replace("https://dblp.org/rec/", "")
                elif "key" in pub:
                    dblp_key = pub.get("key", "").replace("dblp:", "")
                else:
                    dblp_key = pub.get("@id", "").replace("dblp:", "")

                result = {
                    "title": info.get("title", ""),
                    "authors": authors,
                    "venue": info.get("venue", ""),
                    "year": int(info.get("year", 0)) if info.get("year") else None,
                    "type": info.get("type", ""),
                    "doi": info.get("doi", ""),
                    "ee": info.get("ee", ""),
                    "url": info.get("url", ""),
                    "dblp_key": dblp_key,  # Use more specific name for the DBLP key
                }
                results.append(result)
    except requests.exceptions.Timeout:
        logger.error(f"Timeout error searching DBLP after {REQUEST_TIMEOUT} seconds")
        # Provide timeout error information
        timeout_msg = f"ERROR: Query '{single_query}' timed out after {REQUEST_TIMEOUT} seconds"
        results.append(
            {
                "title": timeout_msg,
                "authors": [],
                "venue": "Error",
                "year": None,
                "error": f"Timeout after {REQUEST_TIMEOUT} seconds",
            }
        )
    except Exception as e:
        logger.error(f"Error searching DBLP: {e}")
        # Return error result instead of mock data
        error_msg = f"ERROR: DBLP API error for query '{single_query}': {str(e)}"
        results.append(
            {
                "title": error_msg,
                "authors": [],
                "venue": "Error",
                "year": None,
                "error": str(e),
            }
        )
    return results


def search(
    query: str,
    max_results: int = 10,
    year_from: int | None = None,
    year_to: int | None = None,
    venue_filter: str | None = None,
    include_bibtex: bool = False,
) -> list[dict[str, Any]]:
    """
    Search DBLP using their public API.

    Parameters:
        query (str): The search query string.
        max_results (int, optional): Maximum number of results to return. Default is 10.
        year_from (int, optional): Lower bound for publication year.
        year_to (int, optional): Upper bound for publication year.
        venue_filter (str, optional): Case-insensitive substring filter
            for publication venues.
        include_bibtex (bool, optional): Whether to include BibTeX entries
            in the results. Default is False.

    Returns:
        List[Dict[str, Any]]: A list of publication dictionaries.
    """
    query_lower = query.lower()
    if "(" in query or ")" in query:
        logger.warning(
            "Parentheses are not supported in boolean queries. "
            "They will be treated as literal characters."
        )
    results = []
    if " or " in query_lower:
        subqueries = [q.strip() for q in query_lower.split(" or ") if q.strip()]
        seen = set()
        for q in subqueries:
            for pub in _fetch_publications(q, max_results):
                identifier = (pub.get("title"), pub.get("year"))
                if identifier not in seen:
                    results.append(pub)
                    seen.add(identifier)
    else:
        results = _fetch_publications(query, max_results)

    filtered_results = []
    for result in results:
        if year_from or year_to:
            year = result.get("year")
            if year:
                try:
                    year = int(year)
                    if (year_from and year < year_from) or (year_to and year > year_to):
                        continue
                except (ValueError, TypeError):
                    pass
        if venue_filter:
            venue = result.get("venue", "")
            if venue_filter.lower() not in venue.lower():
                continue
        filtered_results.append(result)

    if not filtered_results:
        logger.info("No results found. Consider revising your query syntax.")

    filtered_results = filtered_results[:max_results]

    # Fetch BibTeX entries if requested
    if include_bibtex:
        for result in filtered_results:
            if "dblp_key" in result and result["dblp_key"]:
                result["bibtex"] = fetch_bibtex_entry(result["dblp_key"])

    return filtered_results


def get_author_publications(
    author_name: str,
    similarity_threshold: float,
    max_results: int = 20,
    include_bibtex: bool = False,
) -> dict[str, Any]:
    """
    Get publication information for a specific author with fuzzy matching.

    Parameters:
        author_name (str): Author name to search for.
        similarity_threshold (float): Threshold for fuzzy matching (0-1).
        max_results (int, optional): Maximum number of results to return. Default is 20.
        include_bibtex (bool, optional): Whether to include BibTeX entries. Default is False.

    Returns:
        Dict[str, Any]: Dictionary with author publication information.
    """
    logger.info(
        f"Getting publications for author: {author_name} with similarity threshold {similarity_threshold}"
    )
    author_query = f"author:{author_name}"
    publications = search(author_query, max_results=max_results * 2)

    filtered_publications = []
    for pub in publications:
        best_ratio = 0.0
        for candidate in pub.get("authors", []):
            ratio = difflib.SequenceMatcher(None, author_name.lower(), candidate.lower()).ratio()
            if ratio > best_ratio:
                best_ratio = ratio
        if best_ratio >= similarity_threshold:
            filtered_publications.append(pub)

    filtered_publications = filtered_publications[:max_results]

    # Fetch BibTeX entries if requested
    if include_bibtex:
        for pub in filtered_publications:
            if "dblp_key" in pub and pub["dblp_key"]:
                pub["bibtex"] = fetch_bibtex_entry(pub["dblp_key"])

    venues = Counter([p.get("venue", "") for p in filtered_publications])
    years = Counter([p.get("year", "") for p in filtered_publications])
    types = Counter([p.get("type", "") for p in filtered_publications])

    return {
        "name": author_name,
        "publication_count": len(filtered_publications),
        "publications": filtered_publications,
        "stats": {
            "venues": venues.most_common(5),
            "years": years.most_common(5),
            "types": dict(types),
        },
    }


def fuzzy_title_search(
    title: str,
    similarity_threshold: float,
    max_results: int = 10,
    year_from: int | None = None,
    year_to: int | None = None,
    venue_filter: str | None = None,
    include_bibtex: bool = False,
) -> list[dict[str, Any]]:
    """
    Search DBLP for publications with fuzzy title matching.

    Uses multiple search strategies to improve recall:
    1. Search with "title:" prefix
    2. Search without prefix (broader matching)
    3. Calculate similarity scores and rank by best match

    Note: DBLP's search ranking may not prioritize the exact paper you're looking for.
    For best results, include author name or year in the title parameter
    (e.g., "Attention is All You Need Vaswani" or use the regular search() function).

    Parameters:
        title (str): Full or partial title of the publication (case-insensitive).
        similarity_threshold (float): A float between 0 and 1 where 1.0 means an exact match.
        max_results (int, optional): Maximum number of publications to return. Default is 10.
        year_from (int, optional): Lower bound for publication year.
        year_to (int, optional): Upper bound for publication year.
        venue_filter (str, optional): Case-insensitive substring filter for publication venues.
        include_bibtex (bool, optional): Whether to include BibTeX entries. Default is False.

    Returns:
        List[Dict[str, Any]]: A list of publication objects sorted by title similarity score.
    """
    logger.info(f"Searching for title: '{title}' with similarity threshold {similarity_threshold}")

    candidates = []
    seen_titles = set()

    # Strategy 1: Search with title prefix
    title_query = f"title:{title}"
    results = search(
        title_query,
        max_results=max_results * 3,
        year_from=year_from,
        year_to=year_to,
        venue_filter=venue_filter,
    )
    for pub in results:
        t = pub.get("title", "")
        if t not in seen_titles:
            candidates.append(pub)
            seen_titles.add(t)

    # Strategy 2: Search without prefix
    results = search(
        title,
        max_results=max_results * 2,
        year_from=year_from,
        year_to=year_to,
        venue_filter=venue_filter,
    )
    for pub in results:
        t = pub.get("title", "")
        if t not in seen_titles:
            candidates.append(pub)
            seen_titles.add(t)

    # Calculate similarity scores
    filtered = []
    for pub in candidates:
        pub_title = pub.get("title", "")
        ratio = difflib.SequenceMatcher(None, title.lower(), pub_title.lower()).ratio()
        if ratio >= similarity_threshold:
            pub["similarity"] = ratio
            filtered.append(pub)

    # Sort by similarity score (highest first)
    filtered = sorted(filtered, key=lambda x: x.get("similarity", 0), reverse=True)

    filtered = filtered[:max_results]

    # Fetch BibTeX entries if requested
    if include_bibtex:
        for pub in filtered:
            if "dblp_key" in pub and pub["dblp_key"]:
                bibtex = fetch_bibtex_entry(pub["dblp_key"])
                if bibtex:
                    pub["bibtex"] = bibtex

    return filtered


def _parse_bibtex_fields(bibtex: str) -> dict[str, str]:
    """Parse BibTeX entry into a dictionary of fields."""
    fields = {}

    # Extract entry type
    type_match = re.match(r"@(\w+)\{", bibtex)
    if type_match:
        fields["_type"] = type_match.group(1).lower()

    # Extract fields using regex - handles multiline values
    # Match field = {value} or field = "value"
    field_pattern = re.compile(r'(\w+)\s*=\s*[{"](.+?)[}"](?=,\s*\w+\s*=|\s*\}$)', re.DOTALL)
    for match in field_pattern.finditer(bibtex):
        field_name = match.group(1).lower()
        field_value = match.group(2).strip()
        fields[field_name] = field_value

    return fields


def _get_venue_abbreviation(venue: str, entry_type: str) -> str:
    """Extract or generate venue abbreviation from venue name."""
    if not venue:
        return "MISC"

    # Clean up venue string (remove braces and extra spaces)
    venue_clean = re.sub(r'[{}]', '', venue).strip()

    # Venue abbreviations - ordered list for priority matching (more specific first)
    venue_abbrevs_ordered = [
        # Visualization - more specific first
        ("pacific visualization symposium", "PacificVIS"),
        ("pacific visualization", "PacificVIS"),
        ("pacificvis", "PacificVIS"),
        ("pacific vis", "PacificVIS"),
        ("ieee trans. vis. comput. graph", "TVCG"),
        ("ieee transactions on visualization and computer graphics", "TVCG"),
        ("tvcg", "TVCG"),
        ("ieee visualization conference", "VIS"),
        ("ieee vis", "VIS"),
        ("eurographics conference on visualization", "EuroVIS"),
        ("eurovis", "EuroVIS"),
        ("comput. graph. forum", "CGF"),
        ("computer graphics forum", "CGF"),
        ("cgf", "CGF"),
        ("comput. graph.", "CG"),
        ("computers and graphics", "CG"),
        ("computers & graphics", "CG"),
        # Graphics
        ("acm trans. graph", "TOG"),
        ("acm transactions on graphics", "TOG"),
        ("siggraph asia", "SIGGRAPHAsia"),
        ("acm siggraph", "SIGGRAPH"),
        ("siggraph", "SIGGRAPH"),
        # HCI
        ("chi conference", "CHI"),
        ("acm chi", "CHI"),
        ("uist", "UIST"),
        # ML/AI
        ("neurips", "NeurIPS"),
        ("neural information processing systems", "NeurIPS"),
        ("nips", "NeurIPS"),
        ("icml", "ICML"),
        ("international conference on machine learning", "ICML"),
        ("iclr", "ICLR"),
        ("international conference on learning representations", "ICLR"),
        ("aaai", "AAAI"),
        ("ijcai", "IJCAI"),
        # CV
        ("cvpr", "CVPR"),
        ("computer vision and pattern recognition", "CVPR"),
        ("iccv", "ICCV"),
        ("international conference on computer vision", "ICCV"),
        ("eccv", "ECCV"),
        ("european conference on computer vision", "ECCV"),
        # NLP
        ("emnlp", "EMNLP"),
        ("naacl", "NAACL"),
        ("coling", "COLING"),
        ("annual meeting of the association for computational linguistics", "ACL"),
        (" acl ", "ACL"),
        # arXiv
        ("arxiv", "arXiv"),
        ("corr", "arXiv"),
    ]

    venue_lower = venue_clean.lower()

    # Check for known abbreviations using ordered list (more specific patterns first)
    for key, abbrev in venue_abbrevs_ordered:
        if key in venue_lower:
            return abbrev

    # Try to extract abbreviation from venue name
    # Look for uppercase abbreviations in the venue string
    abbrev_match = re.search(r'\b([A-Z]{2,})\b', venue_clean)
    if abbrev_match:
        return abbrev_match.group(1)

    # Generate abbreviation from first letters of words
    words = re.findall(r'[A-Z][a-z]*', venue_clean)
    if words:
        return ''.join(w[0] for w in words[:4]).upper()

    # Fallback: use first 4 characters
    return venue_clean[:4].upper() if venue_clean else "MISC"


def _generate_citation_key(first_author: str, venue: str, year: str, entry_type: str) -> str:
    """Generate citation key in format: FirstAuthorLastName-VenueAbbrevYY."""
    # Extract last name of first author
    if not first_author:
        last_name = "Unknown"
    else:
        # Handle "Last, First" format
        if "," in first_author:
            last_name = first_author.split(",")[0].strip()
        else:
            # Handle "First Last" format - take last word
            parts = first_author.strip().split()
            last_name = parts[-1] if parts else "Unknown"

        # Clean up the last name (remove special chars, keep only letters)
        last_name = re.sub(r'[^a-zA-Z]', '', last_name)
        if not last_name:
            last_name = "Unknown"

    # Get venue abbreviation
    venue_abbrev = _get_venue_abbreviation(venue, entry_type)

    # Get last two digits of year
    year_short = year[-2:] if year and len(year) >= 2 else "00"

    return f"{last_name}-{venue_abbrev}{year_short}"


def _normalize_venue_name(venue: str, is_journal: bool) -> str:
    """
    Normalize venue name to standard format.
    For journals: full official name
    For conferences: "Proceedings of <Conference Name>" without location/edition
    """
    if not venue:
        return ""

    # Clean up venue string (remove braces and extra whitespace)
    venue_clean = re.sub(r'[{}]', '', venue)
    venue_clean = re.sub(r'\s+', ' ', venue_clean).strip()
    venue_lower = venue_clean.lower()

    # Standard journal names mapping - ordered for priority matching
    journal_names_ordered = [
        # Visualization & Graphics
        ("ieee trans. vis. comput. graph", "IEEE Transactions on Visualization and Computer Graphics"),
        ("ieee transactions on visualization and computer graphics", "IEEE Transactions on Visualization and Computer Graphics"),
        ("tvcg", "IEEE Transactions on Visualization and Computer Graphics"),
        ("comput. graph. forum", "Computer Graphics Forum"),
        ("computer graphics forum", "Computer Graphics Forum"),
        ("cgf", "Computer Graphics Forum"),
        ("comput. graph.", "Computers and Graphics"),
        ("computers and graphics", "Computers and Graphics"),
        ("computers & graphics", "Computers and Graphics"),
        ("c&g", "Computers and Graphics"),
        ("acm trans. graph", "ACM Transactions on Graphics"),
        ("acm transactions on graphics", "ACM Transactions on Graphics"),
        ("tog", "ACM Transactions on Graphics"),
        ("ieee computer graphics and applications", "IEEE Computer Graphics and Applications"),
        ("ieee cg&a", "IEEE Computer Graphics and Applications"),
        ("cg&a", "IEEE Computer Graphics and Applications"),
        # ML/AI
        ("journal of machine learning research", "Journal of Machine Learning Research"),
        ("j. mach. learn. res", "Journal of Machine Learning Research"),
        ("jmlr", "Journal of Machine Learning Research"),
        # General
        ("nature", "Nature"),
        ("science", "Science"),
        ("corr", "CoRR"),
        ("arxiv", "arXiv"),
    ]

    # Standard conference names mapping (booktitle format)
    # Note: Order matters for matching - more specific patterns should be checked first
    conference_names_ordered = [
        # Visualization - more specific first
        ("pacific visualization symposium", "Proceedings of IEEE Pacific Visualization Symposium"),
        ("pacific visualization", "Proceedings of IEEE Pacific Visualization Symposium"),
        ("pacificvis", "Proceedings of IEEE Pacific Visualization Symposium"),
        ("pacific vis", "Proceedings of IEEE Pacific Visualization Symposium"),
        ("eurovis", "Proceedings of Eurographics Conference on Visualization"),
        ("eurographics conference on visualization", "Proceedings of Eurographics Conference on Visualization"),
        ("ieee vis", "Proceedings of IEEE Visualization Conference"),
        ("ieee visualization conference", "Proceedings of IEEE Visualization Conference"),
        # Graphics
        ("siggraph asia", "Proceedings of ACM SIGGRAPH Asia"),
        ("siggraph", "Proceedings of ACM SIGGRAPH"),
        ("acm siggraph", "Proceedings of ACM SIGGRAPH"),
        # HCI
        ("chi conference", "Proceedings of ACM CHI Conference on Human Factors in Computing Systems"),
        ("acm chi", "Proceedings of ACM CHI Conference on Human Factors in Computing Systems"),
        ("uist", "Proceedings of ACM Symposium on User Interface Software and Technology"),
        # ML/AI
        ("neurips", "Proceedings of Advances in Neural Information Processing Systems"),
        ("nips", "Proceedings of Advances in Neural Information Processing Systems"),
        ("neural information processing systems", "Proceedings of Advances in Neural Information Processing Systems"),
        ("icml", "Proceedings of International Conference on Machine Learning"),
        ("international conference on machine learning", "Proceedings of International Conference on Machine Learning"),
        ("iclr", "Proceedings of International Conference on Learning Representations"),
        ("international conference on learning representations", "Proceedings of International Conference on Learning Representations"),
        ("aaai", "Proceedings of Association for the Advancement of Artificial Intelligence"),
        ("ijcai", "Proceedings of International Joint Conference on Artificial Intelligence"),
        # CV
        ("cvpr", "Proceedings of IEEE/CVF Conference on Computer Vision and Pattern Recognition"),
        ("computer vision and pattern recognition", "Proceedings of IEEE/CVF Conference on Computer Vision and Pattern Recognition"),
        ("iccv", "Proceedings of IEEE/CVF International Conference on Computer Vision"),
        ("international conference on computer vision", "Proceedings of IEEE/CVF International Conference on Computer Vision"),
        ("eccv", "Proceedings of European Conference on Computer Vision"),
        ("european conference on computer vision", "Proceedings of European Conference on Computer Vision"),
        # NLP
        ("emnlp", "Proceedings of Conference on Empirical Methods in Natural Language Processing"),
        ("naacl", "Proceedings of Annual Conference of the North American Chapter of the ACL"),
        ("coling", "Proceedings of International Conference on Computational Linguistics"),
        ("annual meeting of the association for computational linguistics", "Proceedings of Annual Meeting of the Association for Computational Linguistics"),
        (" acl ", "Proceedings of Annual Meeting of the Association for Computational Linguistics"),
    ]
    if is_journal:
        # Check for known journal names using ordered list
        for key, standard_name in journal_names_ordered:
            if key in venue_lower:
                return standard_name
        # Return cleaned original if not found
        return venue_clean
    else:
        # Check for known conference names using ordered list (more specific patterns first)
        for key, standard_name in conference_names_ordered:
            if key in venue_lower:
                return standard_name
        # For unknown conferences, try to clean up
        # Remove year, location, edition info
        # Pattern: remove things like "2017", "December 4-9", "Long Beach, CA", "30th", etc.
        cleaned = re.sub(r'\d{4}', '', venue_clean)  # Remove years
        cleaned = re.sub(r'\d+(?:st|nd|rd|th)\s+', '', cleaned)  # Remove ordinals like "30th"
        cleaned = re.sub(r',\s*[A-Z][a-z]+(?:\s+\d+[-–]\d+)?(?:,\s*\d{4})?', '', cleaned)  # Remove dates
        cleaned = re.sub(r',\s*[A-Z]{2,}\s*,?\s*\{?[A-Z]+\}?', '', cleaned)  # Remove locations like ", CA, USA"
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        cleaned = cleaned.rstrip(',').strip()

        if cleaned and not cleaned.lower().startswith("proceedings"):
            return f"Proceedings of {cleaned}"
        return cleaned if cleaned else venue_clean


def _normalize_title(title: str) -> str:
    """
    Normalize title formatting.
    - Keep acronyms in braces but move punctuation outside
    - e.g., "{SSR-TVD:}" -> "{SSR-TVD}:"
    """
    if not title:
        return ""

    # Clean up whitespace
    title_clean = re.sub(r'\s+', ' ', title).strip()

    # Fix pattern like {ABC:} -> {ABC}:
    title_clean = re.sub(r'\{([^}]+)([:;,.])\}', r'{\1}\2', title_clean)

    # Remove outer braces if the entire title is wrapped
    if title_clean.startswith('{') and title_clean.endswith('}'):
        # Check if these are matching braces
        depth = 0
        is_outer = True
        for i, c in enumerate(title_clean):
            if c == '{':
                depth += 1
            elif c == '}':
                depth -= 1
            if depth == 0 and i < len(title_clean) - 1:
                is_outer = False
                break
        if is_outer:
            title_clean = title_clean[1:-1]

    return title_clean


def _normalize_author(author: str) -> str:
    """Normalize author string - clean up whitespace and formatting."""
    if not author:
        return ""

    # Split by "and" and rejoin cleanly
    authors = re.split(r'\s+and\s+', author)
    authors = [re.sub(r'\s+', ' ', a).strip() for a in authors]
    return " and ".join(authors)


def _format_bibtex_entry(fields: dict[str, str], citation_key: str) -> str:
    """Format BibTeX entry according to the specified format."""
    entry_type = fields.get("_type", "article")

    # Normalize fields
    normalized_fields = dict(fields)

    # Normalize author
    if "author" in normalized_fields:
        normalized_fields["author"] = _normalize_author(normalized_fields["author"])

    # Normalize title
    if "title" in normalized_fields:
        normalized_fields["title"] = _normalize_title(normalized_fields["title"])

    # Determine if it's article or inproceedings based on available fields
    if entry_type in ["article", "journal"]:
        # Article format
        lines = [f"@article{{{citation_key},"]
        field_order = ["author", "title", "journal", "volume", "number", "pages", "year", "doi"]

        # Normalize journal name
        if "journal" in normalized_fields:
            normalized_fields["journal"] = _normalize_venue_name(normalized_fields["journal"], is_journal=True)
    else:
        # Inproceedings format (conference papers)
        lines = [f"@inproceedings{{{citation_key},"]
        field_order = ["author", "title", "booktitle", "year", "pages", "doi"]

        # Normalize booktitle
        if "booktitle" in normalized_fields:
            normalized_fields["booktitle"] = _normalize_venue_name(normalized_fields["booktitle"], is_journal=False)

    for field in field_order:
        value = normalized_fields.get(field, "")
        if value:
            lines.append(f'  {field} = "{value}",')
        # Skip empty optional fields (pages, volume, number, doi)
        elif field not in ["pages", "volume", "number", "doi"]:
            lines.append(f'  {field} = "",')

    lines.append("}")
    return "\n".join(lines)


def _classify_publication_type(dblp_key: str, fields: dict[str, str]) -> tuple[int, str]:
    """
    Classify publication type and return priority.
    Returns: (priority, type_name) where lower priority = more preferred
    Priority: 1 = journal, 2 = conference, 3 = arxiv/other
    """
    key_lower = dblp_key.lower()
    entry_type = fields.get("_type", "").lower()
    journal = fields.get("journal", "").lower()
    booktitle = fields.get("booktitle", "").lower()

    # Check if it's arXiv
    if "arxiv" in key_lower or "corr" in key_lower or "arxiv" in journal:
        return (3, "arxiv")

    # Check if it's a journal article
    if entry_type == "article" or "journals/" in key_lower:
        return (1, "journal")

    # Check if it's a conference paper
    if entry_type == "inproceedings" or "conf/" in key_lower:
        return (2, "conference")

    # Default to other
    return (3, "other")


def _find_best_version(title: str, authors: list[str], year: str) -> str | None:
    """
    Search for the best version of a publication (journal > conference > arxiv).
    Returns the dblp_key of the best version found, or None if search fails.
    """
    if not title:
        return None

    # Search for the paper by title
    search_query = title
    if authors:
        # Add first author to improve search accuracy
        first_author = authors[0].split()[-1] if authors[0] else ""
        if first_author:
            search_query = f"{first_author} {title}"

    try:
        results = search(search_query, max_results=20)

        # Filter results by title similarity and classify by type
        candidates = []
        for result in results:
            result_title = result.get("title", "").lower().strip()
            search_title = title.lower().strip()

            # Check title similarity
            ratio = difflib.SequenceMatcher(None, search_title, result_title).ratio()
            if ratio < 0.7:  # Skip if title is too different
                continue

            dblp_key = result.get("dblp_key", "")
            if not dblp_key:
                continue

            # Classify and get priority
            priority, pub_type = _classify_publication_type(dblp_key, {
                "_type": result.get("type", ""),
                "venue": result.get("venue", ""),
            })

            candidates.append({
                "dblp_key": dblp_key,
                "priority": priority,
                "pub_type": pub_type,
                "similarity": ratio,
            })

        if not candidates:
            return None

        # Sort by priority (lower is better), then by similarity (higher is better)
        candidates.sort(key=lambda x: (x["priority"], -x["similarity"]))

        return candidates[0]["dblp_key"]

    except Exception as e:
        logger.warning(f"Error searching for best version: {e}")
        return None


def fetch_and_process_bibtex(dblp_key: str, custom_citation_key: str | None = None):
    """
    Fetch BibTeX from DBLP and format it according to specified rules.

    Priority: Journal > Conference > arXiv/Other
    Format: Standardized article/inproceedings format
    Citation key: FirstAuthorLastName-VenueAbbrevYY (e.g., Han-VIS22)

    If the given dblp_key points to an arXiv version, this function will
    automatically search for a journal or conference version and use that instead.

    Parameters:
        dblp_key (str): DBLP key for the publication
        custom_citation_key (str, optional): If provided, use this as citation key instead of auto-generating

    Returns:
        str: Formatted BibTeX content, or error message
    """
    try:
        # Clean up dblp_key
        dblp_key = dblp_key.strip()
        if dblp_key.endswith(".bib"):
            dblp_key = dblp_key[:-4]
        if "dblp.org/rec/" in dblp_key:
            dblp_key = dblp_key.split("dblp.org/rec/")[-1]

        url = f"https://dblp.org/rec/{dblp_key}.bib"
        logger.info(f"Fetching BibTeX from: {url}")

        response = requests.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        bibtex = response.text

        if not bibtex or bibtex.isspace():
            return "% Error: Empty BibTeX response"

        # Parse the BibTeX entry
        fields = _parse_bibtex_fields(bibtex)

        # Classify publication type
        priority, pub_type = _classify_publication_type(dblp_key, fields)

        # If this is an arXiv paper, try to find a better version
        if pub_type == "arxiv":
            title = fields.get("title", "")
            author = fields.get("author", "")
            authors = [a.strip() for a in author.split(" and ")] if author else []
            year = fields.get("year", "")

            logger.info(f"Found arXiv version, searching for journal/conference version...")
            better_key = _find_best_version(title, authors, year)

            if better_key and better_key != dblp_key:
                # Check if the better version is actually better
                better_priority, better_type = _classify_publication_type(better_key, {})
                if better_priority < priority:
                    logger.info(f"Found better version: {better_key} ({better_type})")
                    # Recursively fetch the better version
                    return fetch_and_process_bibtex(better_key, custom_citation_key)

        # Extract author information
        author = fields.get("author", "")
        # Split by "and" with optional whitespace/newlines around it
        authors_list = re.split(r'\s+and\s+', author) if author else []
        first_author = authors_list[0].strip() if authors_list else ""

        # Extract year
        year = fields.get("year", "")

        # Determine venue based on publication type
        if pub_type == "journal":
            venue = fields.get("journal", "")
        elif pub_type == "conference":
            venue = fields.get("booktitle", "")
        else:
            venue = fields.get("journal", "") or fields.get("booktitle", "") or "arXiv"

        # Generate or use provided citation key
        if custom_citation_key:
            citation_key = custom_citation_key
        else:
            citation_key = _generate_citation_key(first_author, venue, year, pub_type)

        # Format the BibTeX entry
        formatted_bibtex = _format_bibtex_entry(fields, citation_key)

        return formatted_bibtex

    except requests.exceptions.Timeout:
        logger.error(f"Timeout fetching BibTeX after {REQUEST_TIMEOUT} seconds")
        return f"% Error: Timeout fetching BibTeX after {REQUEST_TIMEOUT} seconds"
    except Exception as e:
        logger.error(f"Error fetching BibTeX: {str(e)}", exc_info=True)
        return f"% Error fetching BibTeX: {str(e)}"


def fetch_bibtex_entry(dblp_key: str) -> str:
    """
    Fetch BibTeX entry from DBLP by key.

    Parameters:
        dblp_key (str): DBLP publication key.

    Returns:
        str: BibTeX entry, or empty string if not found.
    """
    try:
        # Make sure we have a valid key
        if not dblp_key or dblp_key.isspace():
            logger.warning("Empty or invalid DBLP key provided")
            return ""

        # Try multiple URL formats to increase chances of success
        urls_to_try = []

        # Format 1: Direct key
        urls_to_try.append(f"https://dblp.org/rec/{dblp_key}.bib")

        # Format 2: If the key has slashes, it might be a full path
        if "/" in dblp_key:
            urls_to_try.append(f"https://dblp.org/rec/{dblp_key}.bib")

        # Format 3: If the key has a colon, it might be a DBLP-style key
        if ":" in dblp_key:
            clean_key = dblp_key.replace(":", "/")
            urls_to_try.append(f"https://dblp.org/rec/{clean_key}.bib")

        # Try each URL until one works
        for url in urls_to_try:
            logger.info(f"Fetching BibTeX from: {url}")
            response = requests.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT)
            logger.info(f"Response status: {response.status_code}")

            if response.status_code == 200:
                bibtex = response.text
                if not bibtex or bibtex.isspace():
                    logger.warning(f"Received empty BibTeX content for URL: {url}")
                    continue

                logger.info(f"BibTeX content (first 100 chars): {bibtex[:100]}")

                # Extract the citation type and key (e.g., @article{DBLP:journals/jmlr/ChowdheryNDBMGMBCDDRSSTWPLLNSZDYJGKPSN23,)
                citation_key_match = re.match(r"@(\w+){([^,]+),", bibtex)
                if citation_key_match:
                    citation_type = citation_key_match.group(1)
                    old_key = citation_key_match.group(2)
                    logger.info(f"Found citation type: {citation_type}, key: {old_key}")

                    # Create a new key based on the first author's last name and year
                    # Try to extract author and year from the DBLP key or from the BibTeX content
                    author_year_match = re.search(r"([A-Z][a-z]+).*?(\d{2,4})", dblp_key)

                    if author_year_match:
                        author = author_year_match.group(1)
                        year = author_year_match.group(2)
                        if len(year) == 2:  # Convert 2-digit year to 4-digit
                            year = "20" + year if int(year) < 50 else "19" + year
                        new_key = f"{author}{year}"
                        logger.info(f"Generated new key: {new_key}")
                    else:
                        # If we can't extract from key, create a simpler key from the DBLP key
                        parts = dblp_key.split("/")
                        new_key = parts[-1] if parts else dblp_key
                        logger.info(f"Using fallback key: {new_key}")

                    # Replace the old key with the new key
                    bibtex = bibtex.replace(f"{{{old_key},", f"{{{new_key},", 1)
                    logger.info("Replaced old key with new key")

                    return bibtex
                else:
                    logger.warning(
                        f"Could not parse citation key pattern from BibTeX: {bibtex[:100]}..."
                    )
                    return bibtex  # Return the original if we couldn't parse it

        # If we've tried all URLs and none worked
        logger.warning(
            f"Failed to fetch BibTeX for key: {dblp_key} after trying multiple URL formats"
        )
        return ""

    except requests.exceptions.Timeout:
        logger.error(f"Timeout fetching BibTeX for {dblp_key} after {REQUEST_TIMEOUT} seconds")
        return f"% Error: Timeout fetching BibTeX for {dblp_key} after {REQUEST_TIMEOUT} seconds"
    except Exception as e:
        logger.error(f"Error fetching BibTeX for {dblp_key}: {str(e)}", exc_info=True)
        return (
            f"% Error: An unexpected error occurred while fetching BibTeX for {dblp_key}: {str(e)}"
        )


def get_venue_info(venue_name: str) -> dict[str, Any]:
    """
    Get information about a publication venue using DBLP venue search API.
    Returns venue name, acronym, type, and DBLP URL.
    """
    logger.info(f"Getting information for venue: {venue_name}")
    try:
        url = "https://dblp.org/search/venue/api"
        params = {"q": venue_name, "format": "json", "h": 1}
        response = requests.get(url, params=params, headers=HEADERS, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        data = response.json()

        hits = data.get("result", {}).get("hits", {})
        total = int(hits.get("@total", "0"))

        if total > 0:
            hit = hits.get("hit", [])
            if isinstance(hit, list):
                hit = hit[0]

            info = hit.get("info", {})
            return {
                "venue": info.get("venue", ""),
                "acronym": info.get("acronym", ""),
                "type": info.get("type", ""),
                "url": info.get("url", ""),
            }
        else:
            logger.warning(f"No venue found for: {venue_name}")
            return {
                "venue": "",
                "acronym": "",
                "type": "",
                "url": "",
            }
    except Exception as e:
        logger.error(f"Error fetching venue info for {venue_name}: {str(e)}")
        return {
            "venue": "",
            "acronym": "",
            "type": "",
            "url": "",
        }


def calculate_statistics(results: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Calculate statistics from publication results.
    (Documentation omitted for brevity)
    """
    logger.info(f"Calculating statistics for {len(results)} results")
    authors = Counter()
    venues = Counter()
    years = []

    for result in results:
        for author in result.get("authors", []):
            authors[author] += 1

        venue = result.get("venue", "")
        # Handle venue as list or string
        if isinstance(venue, list):
            venue = ", ".join(venue) if venue else ""
        if venue:
            venues[venue] += 1
        else:
            venues["(empty)"] += 1

        year = result.get("year")
        if year:
            with contextlib.suppress(ValueError, TypeError):
                years.append(int(year))

    stats = {
        "total_publications": len(results),
        "time_range": {"min": min(years) if years else None, "max": max(years) if years else None},
        "top_authors": sorted(authors.items(), key=lambda x: x[1], reverse=True),
        "top_venues": sorted(venues.items(), key=lambda x: x[1], reverse=True),
    }

    return stats
