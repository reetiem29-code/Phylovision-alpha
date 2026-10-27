# PhyloVision — Botanical Bioinformatics Workspace

PhyloVision is an educational Streamlit application for exploring already-aligned DNA or protein sequences, pairwise identity, distance-based phylogenetic trees, and sequence differences. This upgrade keeps the existing app workflow and creator credits while refining the interface into a quieter charcoal, sage, and botanical-lilac theme.

## Run locally

1. Use Python 3.10 or newer.
2. Open a terminal in the folder containing `app.py`.
3. Install dependencies and launch:

   ```bash
   python -m pip install -r requirements.txt
   python -m streamlit run app.py
   ```

## Main workflow

1. Open **Start here → Sequence Input** and upload or paste FASTA records.
2. Select DNA or Protein. Sequences must be homologous, already aligned, and equal length for the alignment-based analyses.
3. **Analyze → Alignment Explorer** shows alignment columns.
4. **Analyze → Phylogenetic Tree** builds a Neighbor-Joining or UPGMA tree. Choose the new **Radial garden** layout or the rectangular layout. Optional bootstrap resamples alignment columns.
5. **Explore results → Sequence Similarity** displays pairwise percent identity and exports a CSV. Two or more sequences are supported; pairs with no comparable sites display as unavailable instead of a misleading 0% or 100%.
6. Use **Evolution Lens** to inspect differences, **Understand My Tree** for a plain-language interpretation, and **Creators & About** for project details.

## Important scientific limitations

- This app does not automatically align raw sequences.
- Pairwise identity excludes gaps and ambiguous/missing characters (`-`, `?`, `.`, `N`, `X`) from each pair's denominator.
- Neighbor Joining / UPGMA output depends on the input alignment and distance method. UPGMA assumes a constant evolutionary rate.
- Bootstrap percentages, when calculated, are split-recovery frequencies from resampled alignment columns; they are not probabilities that a clade is true.
- Equal sequence lengths alone do not prove a biologically correct alignment.

## Changes in this upgrade

- Reworked the dashboard to a restrained charcoal + sage botanical palette with quieter cards and clearer hierarchy.
- Reorganized navigation into **Start here**, **Analyze**, **Explore results**, and **Learn & about** groups.
- Fixed pairwise similarity calculation and missing-data handling, with CSV export and explicit no-comparable-sites behavior.
- Added a radial **Radial garden** tree layout while retaining the rectangular view.
- Allowed two-sequence comparisons while keeping tree construction restricted to at least three sequences.
- Kept the existing logo and creator/co-creator section.
