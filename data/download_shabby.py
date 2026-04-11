# =============================================================================
# File: data/download_shabby.py
# Owner: Apoorva Adimulam
#
# Purpose:
#   Downloads and extracts the ShabbyPages dataset (~6,000 clean/degraded
#   image pairs) into data/shabby/. This script is run once before training.
#
# Exports:
#   - download_shabby(dest_dir: str) -> None
#       Downloads the ShabbyPages dataset archive and extracts it to dest_dir.
#       Expected output structure:
#           data/shabby/clean/   — clean reference images
#           data/shabby/degraded/ — degraded counterparts
#
# Dependencies:
#   - requests (or wget / gdown depending on source)
#   - zipfile / tarfile
# =============================================================================
