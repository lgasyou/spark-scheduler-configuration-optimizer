import os

SERIES_NAME = '水浒传'
DIR = '/bigdata/4tb/Series/%s' % SERIES_NAME
for filename in os.listdir(DIR)[:2]:
    new_filename = filename[:4] + 'S01' + filename[4:]
    os.rename(os.path.join(DIR, filename), os.path.join(DIR, new_filename))
