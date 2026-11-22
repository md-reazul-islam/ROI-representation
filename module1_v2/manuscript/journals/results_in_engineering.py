"""Results in Engineering (Elsevier, open access, ISSN 2590-1230).

Confirmed from the journal's Guide for Authors: highlights are 3-5 bullet points of at most 85 characters including
spaces, supplied as a separate editable file; author contributions use CRediT roles. The guide page could not be read in
full (the publisher blocks automated access), so the other entries follow Elsevier's standard requirements; check
`guide_url` before submitting and change only this file if something differs.
"""
PROFILE = dict(
    key='results_in_engineering',
    name='Results in Engineering',
    publisher='Elsevier',
    article_type='Research article',
    guide_url='https://www.sciencedirect.com/journal/results-in-engineering/publish/guide-for-authors',
    verified=['highlights: 3-5 bullets, <= 85 characters each, separate file', 'CRediT author contributions'],
    abstract_max_words=250,
    keywords_max=7,
    highlights=dict(min=3, max=5, max_chars=85),
    research_question_paragraph=False,          # not part of an Elsevier research article
    # citations are renumbered in order of first appearance; style 'elsevier' gives [1], [1,2], [1-3] (en dash)
    citations=dict(order='appearance', style='elsevier'),
    reference_style='elsevier',                 # see references.py
    layout=dict(line_numbers=True, line_spacing=1.5, page_numbers=True),
    # declaration headings, in the order they appear after the Conclusion (keys of submission_info.json 'declarations')
    declarations=[
        ('credit', 'CRediT authorship contribution statement'),
        ('competing', 'Declaration of competing interest'),
        ('funding', 'Funding'),
        ('ethics', 'Ethics statement'),
        ('data', 'Data availability'),
        ('ai', 'Declaration of generative AI and AI-assisted technologies in the writing process'),
        ('acknowledgements', 'Acknowledgements'),
    ],
    cover_letter=dict(salutation='Dear Editor,', scope=(
        'Results in Engineering publishes research across engineering disciplines, including biomedical engineering and '
        'computer vision applied to health monitoring, and values reproducible results and openly released code.')),
    file_prefix='NeoSTAR',
    output_dir='submission_results_in_engineering',
)
