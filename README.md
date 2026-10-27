# PhyloVision — evolutionary sequence workspace

PhyloVision is a student-friendly Streamlit app for exploring aligned DNA or protein sequences, pairwise identity, distance-based phylogenetic trees, and sequence differences.

## Run locally

1. Install Python 3.10 or newer.
2. Open a terminal in this folder (the folder containing `app.py`).
3. Install dependencies and launch:

   ```bash
   python -m pip install -r requirements.txt
   python -m streamlit run app.py
   ```

## Workflow

1. Open **Sequence Input** and upload/paste FASTA.
2. Select DNA or Protein and validate the input.
3. Make sure sequences are homologous, already aligned, and equal length.
4. Use **Alignment Explorer** to inspect columns.
5. Use **Phylogenetic Tree** to build a Neighbor-Joining or UPGMA tree. Optional bootstrap resamples alignment columns; use fewer replicates for quick testing.
6. Use **Evolution Lens** to inspect differences between two sequences.
7. Explore **Understand My Tree** and **Sequence Similarity**.
8. Download the Newick tree, similarity CSV, differences CSV, or PDF report.

## Fixes and additions in this version

- Fixed the tree-building error by explicitly constructing a Biopython `MultipleSeqAlignment` before passing data to `DistanceCalculator.get_distance()`.
- Fixed the built-in teaching example so all five records have equal alignment length.
- Added Neighbor Joining / UPGMA selection, optional column-resampling bootstrap support, Evolution Lens, a compact PDF report, and a basic sequence-quality preview.
- Added a guard against interpreting a tree after the FASTA input has changed.
- Preserved the existing dark botanical-purple styling, existing navigation/features, and creator credits.

## Scientific limitations

- PhyloVision does **not** align raw sequences automatically. Equal lengths alone do not establish that sequences are correctly aligned or homologous.
- Distances are identity-based and intended for education; the tool does not implement advanced DNA/protein substitution models.
- Bootstrap values are percentages of resampled alignments that recovered a split under this pipeline. They are not the probability that a clade is true.
- UPGMA assumes a molecular-clock-like constant rate; Neighbor Joining does not require that assumption.
- Demo records are synthetic teaching examples, not verified reference sequences. Do not use them for scientific claims.

## Credits

- Creator & project lead: Anureet Dutta
- Co-creators: Sreejeeta Kundu, Aishwarya Banik, Arya Ghosh, Ram Argha Nayak
