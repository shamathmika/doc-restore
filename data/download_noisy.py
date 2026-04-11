# =============================================================================
# File: data/download_noisy.py
# Owner: Sakshat Patil
#
# Purpose:
#   Downloads and extracts the NoisyOffice dataset (72 clean/degraded pairs)
#   used for evaluation. This script is run once before eval.
#
# Exports:
#   - download_noisy(dest_dir: str) -> None
#       Downloads the NoisyOffice dataset archive and extracts it to dest_dir.
#       Expected output structure:
#           data/noisy/clean/    — clean reference images
#           data/noisy/degraded/ — degraded counterparts
#
# Dependencies:
#   - requests (or wget / gdown depending on source)
#   - zipfile / tarfile
# =============================================================================
