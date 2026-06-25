#!/usr/bin/env bash
# Fetch the MNIST test split (10k labeled 28x28 digit images) for the
# associative-memory pattern-completion example.
#
# Source: ossci-datasets S3 mirror (stable). IDX ubyte format. Data lands in
# hopfield/data/mnist/ which is gitignored.
set -euo pipefail

DEST="$(cd "$(dirname "$0")/.." && pwd)/data/mnist"
BASE="https://ossci-datasets.s3.amazonaws.com/mnist"

mkdir -p "$DEST"
for f in t10k-images-idx3-ubyte t10k-labels-idx1-ubyte; do
  if [ -f "$DEST/$f" ]; then
    echo "have $f"
  else
    echo "fetching $f.gz"
    curl -sSL --fail -o "$DEST/$f.gz" "$BASE/$f.gz"
    gunzip -f "$DEST/$f.gz"
  fi
done
echo "done -> $DEST"
