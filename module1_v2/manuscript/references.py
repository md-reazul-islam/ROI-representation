"""Reference list of the submission build: structured entries in references.json, rendered in a journal's style.

references.json holds one entry per reference, keyed by `n` (the number used in the manuscript text before
renumbering). Authors, volume, issue, pages, year and DOI of every entry with a DOI were taken from Crossref; titles were
checked against Crossref (and arXiv / Semantic Scholar where Crossref has no record). Citations in the text are
renumbered by build_submission.py; this module only formats entries and in-text citation groups.
"""
import json
import re
from pathlib import Path

DB = {e['n']: e for e in json.load(open(Path(__file__).with_name('references.json'), encoding='utf-8'))}
CITE_GROUP = re.compile(r'\[\d+\](?:\s*(?:,|-|–)\s*\[\d+\])*')


def _initials(given, sep):
    parts = []
    for word in given.split():
        sub = [w for w in word.split('-') if w]
        parts.append('-'.join(w[0] + '.' if not w.endswith('.') else w for w in sub))
    return sep.join(parts).replace('..', '.')


def _names(e, sep, last_and):
    names = [f"{_initials(g, sep)} {f}".strip() for g, f in e['authors']]
    if last_and and len(names) > 1:
        return (', '.join(names[:-1]) + (',' if len(names) > 2 else '') + ' and ' + names[-1])
    return ', '.join(names)


def _pages(e, dash):
    if e.get('pages'):
        return e['pages'].replace('-', dash)
    return ''


def elsevier(e):
    a = _names(e, '', last_and=False)
    t, y = e['title'], e.get('year')
    pages = _pages(e, '–')
    doi = f" https://doi.org/{e['doi']}." if e.get('doi') else ''
    typ = e['type']
    if typ == 'journal':
        loc = pages or e.get('article', '')
        vol = e.get('volume', '')
        s = f"{a}, {t}, {e['venue_abbrev']} {vol} ({y}) {loc}".rstrip() + '.'
    elif typ == 'conference':
        loc = f', pp. {pages}' if pages else (f", Art. {e['article']}" if e.get('article') else '')
        vol = f", vol. {e['volume']}" if e.get('volume') else ''
        note = f", {e['note']}" if e.get('note') else ''
        s = f"{a}, {t}, in: {e['venue_abbrev']}{vol}, {y}{loc}{note}."
    elif typ == 'chapter':
        s = f"{a}, {t}, in: {e['venue']}, {e['publisher']}, {y}, pp. {pages}."
    elif typ == 'book':
        s = f"{a}, {t}, {e['publisher']}, {y}."
    elif typ == 'arxiv':
        s = f"{a}, {t}, {e['venue']} ({y})."
    elif typ == 'software':
        s = f"{a}, {t} ({e['note']}), {y}. {e['url']}."
    else:
        raise ValueError(typ)
    return s + doi


def ieee(e):
    a = _names(e, ' ', last_and=True)
    t, y = e['title'], e.get('year')
    pages = _pages(e, '-')
    doi = f", doi: {e['doi']}" if e.get('doi') else ''
    typ = e['type']
    vol = f", vol. {e['volume']}" if e.get('volume') else ''
    no = f", no. {e['issue']}" if e.get('issue') else ''
    loc = f', pp. {pages}' if pages else (f", Art. no. {e['article']}" if e.get('article') else '')
    if typ == 'journal':
        s = f'{a}, "{t}," {e["venue"]}{vol}{no}{loc}, {y}{doi}.'
    elif typ == 'conference':
        note = f", {e['note']}" if e.get('note') else ''
        s = f'{a}, "{t}," in {e["venue_abbrev"]}{vol}, {y}{loc}{note}{doi}.'
    elif typ == 'chapter':
        s = f'{a}, "{t}," in {e["venue"]}. {e["publisher"]}, {y}{loc}.'
    elif typ == 'book':
        s = f'{a}, {t}. {e["publisher"]}, {y}.'
    elif typ == 'arxiv':
        s = f'{a}, "{t}," {e["venue"]}, {y}.'
    elif typ == 'software':
        s = f'{a}, "{t}," {e["note"]}, {y}. [Online]. Available: {e["url"]}'
    else:
        raise ValueError(typ)
    return s


STYLES = {'elsevier': elsevier, 'ieee': ieee}


def numbers_in(group):
    """'[4]-[10], [13]' -> [4, 5, ..., 10, 13] in reading order."""
    out = []
    for part in re.split(r'\s*,\s*', group):
        m = re.fullmatch(r'\[(\d+)\]\s*[-–]\s*\[(\d+)\]', part)
        if m:
            out += list(range(int(m.group(1)), int(m.group(2)) + 1))
        else:
            out.append(int(part.strip()[1:-1]))
    return out


def format_group(nums, style):
    nums = sorted(set(nums))
    runs, start = [], nums[0]
    for prev, cur in zip(nums, nums[1:] + [None]):
        if cur != prev + 1:
            runs.append((start, prev))
            start = cur
    if style == 'elsevier':
        parts = [f'{a}' if a == b else (f'{a},{b}' if b == a + 1 else f'{a}–{b}') for a, b in runs]
        return '[' + ','.join(parts) + ']'
    parts = [f'[{a}]' if a == b else (f'[{a}], [{b}]' if b == a + 1 else f'[{a}]-[{b}]') for a, b in runs]
    return ', '.join(parts)
