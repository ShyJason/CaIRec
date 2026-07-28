# GitHub Release publication checklist

Use tag `v1.0-assets` and copy `RELEASE_NOTES.md` into the release description.
Upload every file from the prepared `v1.0-assets` asset directory:

- `cair-data-clothing-v1.tar.gz`
- `cair-data-beauty-v1.tar.gz`
- `cair-data-sports-v1.tar.gz`
- `cair-projection-checkpoints-v1.tar.gz`
- `SHA256SUMS`

Before selecting **Publish release**:

1. Confirm the redistribution terms and required attribution for every
   interaction dataset and pre-extracted feature file.
2. In particular, record the provenance of the processed Beauty features; the
   current project history does not identify their original public download.
3. Compare the five uploaded filenames and sizes with the local release
   directory.
4. Verify the archives from that directory:

   ```bash
   sha256sum -c SHA256SUMS
   ```

5. Publish the release, then test a clean download:

   ```bash
   python scripts/download_assets.py --datasets all
   ```

The Clothing and Sports assets follow the MMRec benchmark data layout used by
I3-MRec and DGMRec. Keep the corresponding paper and dataset acknowledgements
in the public release description.
