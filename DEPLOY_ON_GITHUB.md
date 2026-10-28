# Update the existing PhyloVision deployment

This ZIP contains the project source; it does not push changes to GitHub or alter the live website automatically.

1. Download and extract `PhyloVision_Final_Consolidated.zip`.
2. Open your existing repository: https://github.com/reetiem29-code/prime-Phylovision
3. Upload/replace the root-level `app.py`, `requirements.txt`, `README.md`, `phyloVision_logo.png`, and `DEPLOY_ON_GITHUB.md`.
4. Create the `.streamlit` folder in the repository if it does not exist, then upload `.streamlit/config.toml` into it. The leading dot in `.streamlit` matters.
5. Commit the changes to the branch your Streamlit Community Cloud app deploys from (usually `main`).
6. Open Streamlit Community Cloud and check the app's **Manage app / Logs** area if it does not rebuild automatically. If needed, use **Reboot app** after confirming the commit is on the configured branch.
7. Smoke-test the live app: Home → Load demo data → Start; Sequence Input validation; Alignment Explorer blocks; Sequence Similarity; build a tree with bootstrap set to 100; Accession → FASTA with one known accession.

## Important deployment notes

- The plant and mammal teaching examples fetch their real GenBank/RefSeq sequences from NCBI at runtime, so those buttons require network access from the deployed app.
- NCBI accession retrieval does not automatically align arbitrary records. The bundled teaching examples do a simple reference-guided alignment; research-grade analyses should use a curated multiple-sequence alignment.
- If the app reports a dependency error, wait for Community Cloud to finish installing the versions in `requirements.txt`, then inspect the logs.
