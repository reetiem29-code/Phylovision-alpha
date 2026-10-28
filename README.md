# PhyloVision — From Sequences to Evolution

PhyloVision is a student-friendly Streamlit workspace for sequence input, NCBI accession-to-FASTA retrieval, alignment inspection, pairwise sequence identity, distance-based phylogenetic trees, and plain-language interpretation.

## What is included

- **Consistent light theme:** lavender/purple accents, readable widget labels, visible primary actions, and white Plotly chart templates.
- **Guided Lab + Tool Dock:** a step-by-step beginner route, direct tool access, and Quick Action Search in one navigation system.
- **Sequence Input:** FASTA upload/paste with immediate validation feedback, plus a prominent plant teaching-example action.
- **Accession → FASTA:** fetch NCBI nucleotide or protein records, preview and download FASTA, then import directly into Sequence Input. The contact email is kept only in the active Streamlit session.
- **Alignment Explorer:** 60-position blocks, colour-blind-friendly nucleotide colours, conservation markers, and a consensus row.
- **Sequence Similarity:** pairwise identity heatmap with a data-adaptive colour range, mean/minimum/maximum identity, most/least similar pairs, tie reporting, and CSV export.
- **Phylogenetic Tree:** Neighbor Joining or UPGMA, rectangular/radial layouts, Newick export, PDF report, and optional column-resampling bootstrap support (100 replicates by default).
- **Progress:** reflects whether input exists, sequences have equal length for the alignment-based tools, and a tree exists for the current input.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

The theme is configured in `.streamlit/config.toml`.

## Deploy to Streamlit Community Cloud

Upload/commit the contents of this folder to the same GitHub repository used by the existing app. Keep `app.py` at the repository root (or update the Streamlit entry-point setting if you choose a different layout). Streamlit Community Cloud should rebuild after the commit. This ZIP does not update the live deployment by itself.

## Plant teaching dataset and provenance

The teaching examples fetch real marker records from NCBI at runtime. Choose either plant **rbcL** or mammalian mitochondrial **COI**:

- *Arabidopsis thaliana* — [AY174633.1](https://www.ncbi.nlm.nih.gov/nuccore/AY174633.1)
- *Aethionema grandiflora* — [AY167983.1](https://www.ncbi.nlm.nih.gov/nuccore/AY167983.1)
- *Arabis glabra* — [DQ310542.1](https://www.ncbi.nlm.nih.gov/nuccore/DQ310542.1)
- *Berteroa incana* — [KM360667.1](https://www.ncbi.nlm.nih.gov/nuccore/KM360667.1)
- *Anchonium elichrysifolium* — [FN594834.1](https://www.ncbi.nlm.nih.gov/nuccore/FN594834.1)

The five plant accessions are drawn from the published Brassicaceae rbcL barcode dataset in [Decision Tree Algorithm–Generated Single-Nucleotide Polymorphism Barcodes of rbcL Genes for 38 Brassicaceae Species Tagging](https://pmc.ncbi.nlm.nih.gov/articles/PMC5846911/). The mammal example fetches mitochondrial RefSeq records and extracts a 450-nt COI region using the annotated coordinates listed by NCBI: [human NC_012920.1](https://www.ncbi.nlm.nih.gov/gene/004512), [chimpanzee NC_001643.1](https://www.ncbi.nlm.nih.gov/gene/807866), [gorilla NC_011120.1](https://www.ncbi.nlm.nih.gov/gene/6742689), [mouse NC_005089.1](https://www.ncbi.nlm.nih.gov/gene/17708), and [rat NC_001665.2](https://www.ncbi.nlm.nih.gov/gene/26195). The app fetches the actual records, aligns them to a common reference region using a teaching-grade star alignment, and retains the first 450 reference positions (plus any insertion columns). This is provided for classroom exploration, not as a substitute for a curated MAFFT/MUSCLE alignment or expert review. The accessions and retrieval method are visible in the app.

## Scientific notes and limitations

- NCBI retrieval does **not** automatically align arbitrary records. User-supplied sequences must already be aligned and equal length before alignment-dependent tree analysis.
- The plant rbcL and mammal COI examples are transparent, reproducible convenience workflows; inspect the alignment and data provenance before interpreting the tree.
- Pairwise identity excludes gaps and ambiguous/missing symbols from the comparable-position denominator.
- The tree uses an identity-based distance calculation (uncorrected p-distance). UPGMA assumes a constant evolutionary rate; Neighbor Joining does not impose that clock assumption.
- Bootstrap percentages are calculated only when enabled by column resampling. They indicate how often a split recurs under that procedure; they are not the probability that a clade is true.
- Equal sequence lengths do not prove biological homology or a correct alignment. Document accession IDs, marker, alignment method, and settings for research work.
- NCBI fetching requires internet access from the running app and may be affected by temporary service limits.
