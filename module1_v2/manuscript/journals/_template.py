"""Template for a new journal profile: copy to journals/<journal_key>.py and edit. Every field is required.

key                         short name of the file, e.g. 'ieee_access'
name, publisher             shown in the cover letter and the checklist
article_type                e.g. 'Research article'
guide_url                   the journal's guide for authors
verified                    list of requirements actually confirmed from guide_url (the rest are assumptions)
abstract_max_words          the build fails if the abstract is longer
keywords_max                keywords are taken in priority order from build_submission.KEYWORDS
highlights                  dict(min, max, max_chars) or None if the journal has no highlights
research_question_paragraph keep the "Research question:" line under the title (True/False)
citations                   dict(order='appearance' or 'original', style='elsevier' ([1,2], [3-5]) or 'ieee' ([1], [2]-[5]))
reference_style             'elsevier' or 'ieee' (renderers in references.py)
layout                      dict(line_numbers=bool, line_spacing=float, page_numbers=bool)
declarations                list of (key in submission_info.json 'declarations', heading)
cover_letter                dict(salutation, scope = one sentence on why the paper fits the journal)
file_prefix, output_dir     names of the output files and folder (inside module1_v2/reports/<name>/)
"""
PROFILE = None
