import argparse
import json
from pathlib import Path

from worldpolicy.cli import run_training

parser = argparse.ArgumentParser()
parser.add_argument("--config", type=Path, required=True)
print(json.dumps(run_training("diffusion", parser.parse_args().config), indent=2))
