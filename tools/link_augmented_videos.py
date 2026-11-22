"""Collect the DataAugment videos into the folders the HR evaluation reads (from the original HR_all notebook, cell 10).

dataaugment/hr_eval/videos       9 videos: motion, occlusion, lighting x small/medium/large
dataaugment/hr_eval/videos_fast  3 videos: fast motion small/medium/large
dataaugment/hr_eval/videos_fast_xl 1 video: fast motion extra-large
"""
import os
import sys

from common import ROOT

DA = ROOT/'dataaugment'
SETS = {'videos': [DA/'data'/c/s/f'{subj}_{c}_{s}.mkv' for c, subj in (('motion', 'subject1'), ('occlusion', 'subject3'),
                                                                     ('lighting', 'subject4'))
                   for s in ('small', 'medium', 'large')],
        'videos_fast': [DA/'data_motion_fast/motion_fast'/s/f'subject1_motion_fast_{s}.mkv' for s in ('small', 'medium', 'large')],
        'videos_fast_xl': [DA/'data_motion_fast_xl/motion_fast/xlarge/subject1_motion_fast_xlarge.mkv']}


def main():
    missing = [str(f.relative_to(ROOT)) for fs in SETS.values() for f in fs if not f.is_file()]
    if missing:
        sys.exit(f'missing DataAugment videos (run Step B3 first): {missing}')
    for name, files in SETS.items():
        d = DA/'hr_eval'/name
        d.mkdir(exist_ok=True)
        for f in files:
            dst = d/f.name
            if not dst.exists():
                try:
                    os.link(f, dst)                      # hard link, as in the original run
                except OSError:
                    dst.symlink_to(f.resolve())          # different file system: symbolic link
        print(f'{name}: {len(files)} videos')


if __name__ == '__main__':
    main()
