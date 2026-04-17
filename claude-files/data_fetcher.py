"""
data_fetcher.py
---------------
Fetches real Bitcoin block header data from blockchain.info API.
Saves to data/blocks.json for use in training.

Usage:
    python data_fetcher.py --blocks 10000 --output data/blocks.json
"""

import requests
import json
import time
import argparse
import os
from pathlib import Path


API_BASE = "https://blockchain.info"
RATE_LIMIT_DELAY = 0.2  # seconds between requests to be polite


class BlockFetcher:
    def __init__(self, output_path: str = "data/blocks.json"):
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.blocks = []

    def fetch_block(self, block_hash: str) -> dict | None:
        """Fetch a single raw block by hash."""
        try:
            r = requests.get(f"{API_BASE}/rawblock/{block_hash}", timeout=10)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            print(f"  [warn] Failed to fetch block {block_hash[:12]}...: {e}")
            return None

    def fetch_latest_hash(self) -> str:
        """Get the hash of the most recent block."""
        r = requests.get(f"{API_BASE}/latestblock", timeout=10)
        r.raise_for_status()
        return r.json()["hash"]

    def build_chain(self, n_blocks: int = 1000, resume: bool = True):
        """
        Walk backwards from the latest block, collecting n_blocks of data.
        If resume=True and output file exists, load existing and extend.
        """
        # Resume from existing data
        if resume and self.output_path.exists():
            print(f"Loading existing data from {self.output_path}...")
            with open(self.output_path) as f:
                self.blocks = json.load(f)["blocks"]
            print(f"  Loaded {len(self.blocks)} existing blocks")

            if len(self.blocks) >= n_blocks:
                print(f"Already have {len(self.blocks)} blocks, no fetch needed.")
                return

            # Continue from the oldest block we have
            start_hash = self.blocks[-1]["prev_block"]
            remaining = n_blocks - len(self.blocks)
            print(f"Fetching {remaining} more blocks...")
        else:
            start_hash = self.fetch_latest_hash()
            remaining = n_blocks
            print(f"Fetching {n_blocks} blocks from latest...")

        fetched = 0
        current_hash = start_hash

        while fetched < remaining:
            data = self.fetch_block(current_hash)
            if data is None:
                print(f"  Retrying in 2s...")
                time.sleep(2)
                continue

            block_record = {
                "height":      data.get("height", 0),
                "hash":        data["hash"],
                "prev_block":  data["prev_block"],
                "time":        data["time"],
                "nonce":       data["nonce"],
                "bits":        data["bits"],
                "version":     data["ver"],
                "merkle_root": data["mrkl_root"],
                "n_tx":        data["n_tx"],
                "size":        data["size"],
            }

            self.blocks.append(block_record)
            current_hash = data["prev_block"]
            fetched += 1

            if fetched % 100 == 0:
                print(f"  Fetched {fetched}/{remaining} (height={block_record['height']})...")
                self._save()

            time.sleep(RATE_LIMIT_DELAY)

        self._save()
        print(f"\nDone. Total blocks in dataset: {len(self.blocks)}")

    def _save(self):
        with open(self.output_path, "w") as f:
            json.dump({"blocks": self.blocks}, f)

    def load(self) -> list[dict]:
        with open(self.output_path) as f:
            return json.load(f)["blocks"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--blocks", type=int, default=5000, help="Number of blocks to fetch")
    parser.add_argument("--output", type=str, default="data/blocks.json")
    parser.add_argument("--no-resume", action="store_true", help="Start fresh, ignore existing data")
    args = parser.parse_args()

    fetcher = BlockFetcher(output_path=args.output)
    fetcher.build_chain(n_blocks=args.blocks, resume=not args.no_resume)


if __name__ == "__main__":
    main()
