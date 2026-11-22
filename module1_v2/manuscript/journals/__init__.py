"""Journal profiles for build_submission.py.

One file per journal (e.g. results_in_engineering.py) defines PROFILE, a dict with the fields documented in
_template.py. To target another journal, add a profile and run  build_submission.py --journal <file name>.
Nothing else in the build depends on the journal.
"""
import importlib

REQUIRED = {'key', 'name', 'publisher', 'article_type', 'abstract_max_words', 'keywords_max', 'highlights',
            'research_question_paragraph', 'citations', 'reference_style', 'layout', 'declarations', 'cover_letter',
            'file_prefix', 'output_dir', 'guide_url', 'verified'}


def load(name):
    P = importlib.import_module(f'journals.{name}').PROFILE
    missing = REQUIRED - set(P)
    if missing:
        raise SystemExit(f'journal profile {name} lacks {sorted(missing)}')
    return P
