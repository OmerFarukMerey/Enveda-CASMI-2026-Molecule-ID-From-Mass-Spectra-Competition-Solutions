"""Paths shared by this solution's tools. Data lives at the repo root (data/, git-ignored) and mirrors Kaggle's
/kaggle/input layout, so every solution in the repo reads the same download. Override with $CASMI_DATA."""
import os

SOLUTION = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO = os.path.dirname(os.path.dirname(SOLUTION))
DATA = os.environ.get('CASMI_DATA', os.path.join(REPO, 'data'))
INPUT = os.path.join(DATA, 'input')
COMP = os.path.join(INPUT, 'competitions', 'enveda-CASMI26-molecule-id-mass-spectra')
RUNS = os.path.join(SOLUTION, 'runs')
