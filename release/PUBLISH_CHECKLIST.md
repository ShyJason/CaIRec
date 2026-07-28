# GitHub Release publication checklist

Use tag `v1.0-assets` and copy `RELEASE_NOTES.md` into the release description.
Upload every file from the prepared `v1.0-assets` asset directory:

- `cair-missing-payloads-v1.tar.gz`
- `cair-projection-checkpoints-v1.tar.gz`
- `SHA256SUMS`

Before selecting **Publish release**:

1. Confirm that no third-party interaction data, image/text features, or raw
   product content is included.
2. Confirm that `docs/DATASETS.md` still points to the upstream dataset
   sources.
3. Compare the three uploaded filenames and sizes with the local release
   directory.
4. Verify the archives from that directory:

   ```bash
   sha256sum -c SHA256SUMS
   ```

5. Publish the release, then test a clean download:

   ```bash
   python scripts/download_assets.py --payloads all
   ```

Keep the MMRec, I3-MRec, and Amazon Product Data acknowledgements in the public
documentation.
