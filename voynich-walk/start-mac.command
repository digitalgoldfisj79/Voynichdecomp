#!/bin/bash
# Starts a tiny local web server in this folder and opens Voynich Walk in your browser.
# Close this Terminal window to stop it.
cd "$(dirname "$0")"
(sleep 1; open "http://localhost:8000") &
python3 -m http.server 8000
