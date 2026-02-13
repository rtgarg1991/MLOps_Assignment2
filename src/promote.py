from __future__ import annotations

import argparse
from datetime import datetime, timezone

from google.cloud import storage


def copy_prefix(bucket_name: str, source_prefix: str, dest_prefix: str) -> int:
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blobs = list(client.list_blobs(bucket_name, prefix=source_prefix))

    copied = 0
    for blob in blobs:
        if blob.name.endswith("/"):
            continue
        suffix = blob.name[len(source_prefix.rstrip("/")) :].lstrip("/")
        target_name = f"{dest_prefix.rstrip('/')}/{suffix}"
        bucket.copy_blob(blob, bucket, target_name)
        copied += 1
    return copied


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bucket", required=True)
    parser.add_argument("--source-prefix", required=True)
    parser.add_argument("--dest-prefix", default="models/latest")
    args = parser.parse_args()

    copied = copy_prefix(args.bucket, args.source_prefix, args.dest_prefix)
    print(
        f"Promoted {copied} artifacts from {args.source_prefix} to {args.dest_prefix} "
        f"at {datetime.now(timezone.utc).isoformat()}"
    )


if __name__ == "__main__":
    main()
