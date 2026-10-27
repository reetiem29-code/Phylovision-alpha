# PhyloVision — Botanical Bioinformatics

PhyloVision is a Streamlit learning workspace for sequence input, NCBI accession-to-FASTA retrieval, alignment inspection, sequence similarity, and phylogenetic tree exploration.

## Menu update

This version preserves the existing analysis modules and adds a guided-lab menu plus compact direct-access tool dock:

- **Guided Lab:** beginner-friendly steps for preparing sequences, optional accession retrieval, inspecting alignment, building a tree, and exploring findings.
- **Tool Dock:** direct access to modules for experienced users.
- **Quick Action Search:** search tool names and task descriptions, then jump to a matching module.
- **Collapsible sidebar:** use Streamlit's native sidebar collapse/expand control; no custom JavaScript is needed.
- **Readiness indicators:** session-based milestones for sequence input, alignment availability, and tree construction. These indicate workflow state, not biological validity.
- **Visual refinement:** restrained botanical palette, clearer labels, compact navigation buttons, and breathable spacing.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Community Cloud

Commit the updated `app.py`, `README.md`, and `requirements.txt` to the **same GitHub repository** used by the existing app. Streamlit Community Cloud should rebuild the existing deployment after the commit. This archive does not itself update the live website.

## Scientific notes

- Accession retrieval fetches records; it does not by itself perform multiple sequence alignment.
- Confirm that sequences are appropriate and aligned before building a phylogenetic tree.
- The included demo sequences are illustrative teaching data, not verified reference sequences.
- Results depend on input quality and the selected method.
