"""
data_fetcher.py  (v2 — parallel)
---------------------------------
Fetches real Bitcoin block header data from blockchain.info.
Uses concurrent requests to fetch ~20x faster than sequential.

Usage:
    python data_fetcher.py --blocks 10000 --output data/blocks.json
    python data_fetcher.py --blocks 10000 --workers 20   # more parallelism

Can also resume a partially-completed fetch by pointing at the same output file.
"""

import requests
import json
import time
import argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading


API_BASE = "https://blockchain.info"
_print_lock = threading.Lock()


def safe_print(*args, **kwargs):
    with _print_lock:
        print(*args, **kwargs)


def fetch_block(block_hash: str, retries: int = 4) -> dict | None:
    """Fetch a single raw block by hash. Retries on transient failures."""
    for attempt in range(retries):
        try:
            r = requests.get(
                f"{API_BASE}/rawblock/{block_hash}",
                timeout=15,
                headers={"Accept-Encoding": "gzip"},
            )
            if r.status_code == 429:
                wait = 2 ** (attempt + 1)
                safe_print(f"  [rate limit] waiting {wait}s...")
                time.sleep(wait)
                continue
            r.raise_for_status()
            data = r.json()
            return {
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
        except Exception as e:
            if attempt == retries - 1:
                safe_print(f"  [error] {block_hash[:12]}...: {e}")
                return None
            time.sleep(1.5 ** attempt)
    return None


def fetch_hash_at_height(height: int, retries: int = 3) -> tuple[int, str | None]:
    """Fetch the main-chain block hash at a given height."""
    for attempt in range(retries):
        try:
            r = requests.get(
                f"{API_BASE}/block-height/{height}?format=json",
                timeout=15,
            )
            if r.status_code == 429:
                time.sleep(2 ** attempt)
                continue
            r.raise_for_status()
            blocks = r.json().get("blocks", [])
            if blocks:
                main = next((b for b in blocks if b.get("main_chain", True)), blocks[0])
                return height, main["hash"]
        except Exception:
            time.sleep(1)
    return height, None


def get_latest_height() -> int:
    r = requests.get(f"{API_BASE}/latestblock", timeout=10)
    r.raise_for_status()
    return r.json()["height"]


class BlockFetcher:
    def __init__(self, output_path: str = "data/blocks.json", workers: int = 15):
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.workers = workers
        self.blocks: dict[str, dict] = {}  # hash -> block record

    def _load_existing(self):
        if self.output_path.exists():
            with open(self.output_path) as f:
                saved = json.load(f)["blocks"]
            self.blocks = {b["hash"]: b for b in saved}
            safe_print(f"Resuming: loaded {len(self.blocks):,} existing blocks")

    def _save(self):
        sorted_blocks = sorted(self.blocks.values(), key=lambda b: b["height"], reverse=True)
        with open(self.output_path, "w") as f:
            json.dump({"blocks": sorted_blocks}, f)

    def fetch_all(self, n_blocks: int = 10000, resume: bool = True):
        if resume:
            self._load_existing()

        # ── Step 1: Collect the target block hashes by height ─────────────
        latest_height = get_latest_height()
        safe_print(f"Latest Bitcoin height: {latest_height:,}")

        target_heights = list(range(latest_height, latest_height - n_blocks, -1))
        existing_heights = {b["height"] for b in self.blocks.values()}
        heights_to_fetch = [h for h in target_heights if h not in existing_heights]

        safe_print(f"Need hashes for {len(heights_to_fetch):,} heights "
                   f"({len(existing_heights):,} already cached)")

        # Parallel hash-by-height lookup
        height_to_hash: dict[int, str] = {}
        t0 = time.time()

        if heights_to_fetch:
            safe_print(f"Fetching {len(heights_to_fetch):,} block hashes ({self.workers} workers)...")
            with ThreadPoolExecutor(max_workers=self.workers) as pool:
                futures = {pool.submit(fetch_hash_at_height, h): h for h in heights_to_fetch}
                done = 0
                for fut in as_completed(futures):
                    h, bh = fut.result()
                    if bh:
                        height_to_hash[h] = bh
                    done += 1
                    if done % 1000 == 0:
                        safe_print(f"  Hashes: {done:,}/{len(heights_to_fetch):,} "
                                   f"({100*done/len(heights_to_fetch):.0f}%)")

            safe_print(f"  Got {len(height_to_hash):,} hashes in {time.time()-t0:.0f}s")

        # ── Step 2: Fetch full block data for new hashes ───────────────────
        known_hashes = set(self.blocks.keys())
        hashes_to_fetch = [h for h in height_to_hash.values() if h not in known_hashes]

        safe_print(f"\nFetching {len(hashes_to_fetch):,} full blocks ({self.workers} workers)...")
        total = len(hashes_to_fetch)
        done_count = [0]
        t1 = time.time()

        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            futures = {pool.submit(fetch_block, bh): bh for bh in hashes_to_fetch}

            for fut in as_completed(futures):
                result = fut.result()
                if result:
                    self.blocks[result["hash"]] = result

                done_count[0] += 1
                n = done_count[0]

                if n % 500 == 0 or n == total:
                    elapsed = time.time() - t1
                    rate = n / max(elapsed, 0.01)
                    eta = (total - n) / max(rate, 0.01)
                    safe_print(
                        f"  {n:>6,}/{total:,} ({100*n/total:.1f}%) | "
                        f"{rate:.1f} blk/s | ETA {eta/60:.1f} min"
                    )
                    self._save()

        self._save()
        safe_print(f"\nDone. {len(self.blocks):,} blocks saved to {self.output_path}")
        safe_print(f"Total time: {(time.time()-t0)/60:.1f} min")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--blocks",  type=int, default=10000)
    parser.add_argument("--output",  type=str, default="data/blocks.json")
    parser.add_argument("--workers", type=int, default=15,
                        help="Parallel workers. 10-20 is safe. 30+ may get rate-limited.")
    parser.add_argument("--no-resume", action="store_true",
                        help="Start fresh, ignore existing data")
    args = parser.parse_args()

    fetcher = BlockFetcher(output_path=args.output, workers=args.workers)
    fetcher.fetch_all(n_blocks=args.blocks, resume=not args.no_resume)


if __name__ == "__main__":
    main()
