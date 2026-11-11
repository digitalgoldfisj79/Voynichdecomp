#!/usr/bin/env python3
"""Publish completed PIII-CLOSURE outputs without modifying existing Hub files directly."""
import json
import os
import subprocess
from pathlib import Path

from huggingface_hub import HfApi

REPOSITORY = 'Digitalgoldfish79/voynich-dinov3-pipeline'
REMOTE_FOLDER = 'polygraphia_test/PIII_CLOSURE_2026-07-14'
ZIP_PATH = Path('/tmp/PIII_CLOSURE_2026-07-14.zip')

api = HfApi(token=os.environ['HF_TOKEN'])
commit = api.upload_folder(
    folder_path='/tmp/piii_closure',
    path_in_repo=REMOTE_FOLDER + '/files',
    repo_id=REPOSITORY,
    repo_type='dataset',
    commit_message='Add frozen Polygraphia III closure test outputs',
    create_pr=True,
)
zip_commit = api.upload_file(
    path_or_fileobj=str(ZIP_PATH),
    path_in_repo=REMOTE_FOLDER + '/PIII_CLOSURE_2026-07-14.zip',
    repo_id=REPOSITORY,
    repo_type='dataset',
    commit_message='Add Polygraphia III closure reproducibility bundle',
    create_pr=True,
)
print('HF_PR_URL=' + str(getattr(zip_commit, 'pr_url', None) or getattr(commit, 'pr_url', None)))

# Independent temporary export so the completed bundle can be handed back into the chat sandbox.
response = subprocess.check_output(
    ['curl', '-fsS', '-F', 'files[]=@' + str(ZIP_PATH), 'https://uguu.se/upload.php'],
    text=True,
)
print('TEMP_UPLOAD_RESPONSE=' + response.strip())
