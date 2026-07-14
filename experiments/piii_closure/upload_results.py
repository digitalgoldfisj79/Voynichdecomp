#!/usr/bin/env python3
"""Upload completed PIII-CLOSURE outputs without deleting or replacing other files."""
import os
from pathlib import Path

from huggingface_hub import HfApi

REPOSITORY = 'Digitalgoldfish79/voynich-dinov3-pipeline'
REMOTE_FOLDER = 'polygraphia_test/PIII_CLOSURE_2026-07-14'

api = HfApi(token=os.environ['HF_TOKEN'])
api.upload_folder(
    folder_path='/tmp/piii_closure',
    path_in_repo=REMOTE_FOLDER + '/files',
    repo_id=REPOSITORY,
    repo_type='dataset',
    commit_message='Add frozen Polygraphia III closure test outputs',
)
api.upload_file(
    path_or_fileobj='/tmp/PIII_CLOSURE_2026-07-14.zip',
    path_in_repo=REMOTE_FOLDER + '/PIII_CLOSURE_2026-07-14.zip',
    repo_id=REPOSITORY,
    repo_type='dataset',
    commit_message='Add Polygraphia III closure reproducibility bundle',
)
print(
    'ARTIFACT_PATH=https://huggingface.co/datasets/' + REPOSITORY
    + '/resolve/main/' + REMOTE_FOLDER + '/PIII_CLOSURE_2026-07-14.zip'
)
