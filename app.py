from pathlib import Path
import io, re, math
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from Bio import SeqIO, AlignIO, Phylo, Entrez
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio.Align import MultipleSeqAlignment
from Bio.Phylo.TreeConstruction import DistanceCalculator, DistanceTreeConstructor

APP_DIR = Path(__file__).parent
LOGO = APP_DIR / 'phyloVision_logo.png'

st.set_page_config(page_title='PhyloVision | From Sequences to Evolution', page_icon='🧬', layout='wide', initial_sidebar_state='expanded')

st.markdown('''
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap');
:root { --pv-bg:#f7f6fb; --pv-panel:#ffffff; --pv-panel2:#f1eef8; --pv-border:#e4deef; --pv-purple:#8772ad; --pv-purple2:#6e5a94; --pv-lilac:#eee8f7; --pv-text:#292637; --pv-muted:#6e6a7c; }
.stApp { background:radial-gradient(ellipse at 88% 0%, rgba(183,163,218,.15), transparent 34%), linear-gradient(145deg,#faf9fc 0%,#f6f4fa 58%,#f9f8fb 100%); color:var(--pv-text); font-family:'DM Sans',sans-serif; }
[data-testid="stHeader"] { background:transparent; }
[data-testid="stSidebar"] { background:linear-gradient(180deg,#f5f2fa 0%,#f9f8fc 100%); border-right:1px solid #e5dfef; }
[data-testid="stSidebar"] * { color:#393448; }
[data-testid="stSidebar"] [data-testid="stRadio"] label { padding:.45rem .55rem; border-radius:9px; }
[data-testid="stSidebar"] [data-testid="stRadio"] label:hover { background:rgba(135,114,173,.10); }
.block-container { padding-top:1.3rem; padding-bottom:2.5rem; max-width:1480px; }
.pv-card { background:rgba(255,255,255,.88); border:1px solid #e7e1f0; border-radius:15px; padding:19px 21px; margin-bottom:14px; box-shadow:0 7px 24px rgba(65,48,91,.045); }
.pv-muted { color:var(--pv-muted); font-size:.91rem; line-height:1.6; }
.pv-kicker { color:var(--pv-purple2); font-size:.72rem; text-transform:uppercase; letter-spacing:.16em; font-weight:700; }
.pv-title { font-family:'Manrope',sans-serif; font-size:1.65rem; font-weight:700; color:#302a40; margin:.12rem 0 .35rem 0; letter-spacing:-.025em; }
.pv-section { font-family:'Manrope',sans-serif; font-size:1.04rem; font-weight:700; margin-bottom:9px; color:#393148; }
.pv-botanical { color:var(--pv-purple2); }
.pv-pill { display:inline-block; background:#f0ebf8; color:#625080; border:1px solid #ded4ed; border-radius:99px; padding:5px 10px; font-size:.78rem; }
div.stButton > button, div.stDownloadButton > button { border-radius:10px; border:1px solid #d3c7e6; background:#eee8f7; color:#44365f; font-weight:650; min-height:2.5rem; transition:all .16s ease; }
div.stButton > button:hover, div.stDownloadButton > button:hover { border-color:#b6a3d4; background:#e5dcf2; color:#35284f; }
.stTextInput input, .stTextArea textarea, .stSelectbox div[data-baseweb="select"] { background:#fff!important; color:#302a40!important; border-color:#ded8e9!important; border-radius:9px!important; }
[data-testid="stMetric"] { background:#fff; padding:13px 15px; border:1px solid #e7e1f0; border-radius:12px; }
[data-testid="stMetricLabel"] { color:#706a7e!important; }
[data-testid="stMetricValue"] { color:#302a40!important; }
[data-testid="stDataFrame"] { border:1px solid #e7e1f0; border-radius:10px; overflow:hidden; }
hr { border-color:#e5dfef; }
footer {visibility:hidden;}
[data-testid="stSidebar"] { min-width: 285px; }
[data-testid="stSidebar"] .block-container { padding-top:1.1rem; }
.pv-side-label { color:#776396; font-size:.67rem; letter-spacing:.15em; font-weight:800; margin:.35rem 0 .35rem; }
.pv-side-divider { border-top:1px solid #e6e0ef; margin:1rem 0 .8rem; }
.pv-step-hint { color:#797387; font-size:.72rem; line-height:1.35; padding:0 .35rem .42rem 1.55rem; }
.pv-readiness { color:#514a60; font-size:.82rem; line-height:1.9; }
[data-testid="stSidebar"] div.stButton > button { text-align:left; min-height:2.2rem; padding:.35rem .7rem; border-color:#e3dced; background:rgba(255,255,255,.72); color:#443b54; font-weight:600; }
[data-testid="stSidebar"] div.stButton > button[kind="primary"] { background:#e9e1f4; border-color:#cbbce0; color:#44335f; }
[data-testid="stSidebar"] div.stButton > button:hover { background:#eee8f7; border-color:#c7b7de; }
[data-testid="stSidebar"] [data-testid="stTextInput"] input { font-size:.84rem; }
.pv-accession-feature { background:linear-gradient(105deg,#f0eafa 0%,#fff 72%); border:1px solid #ded2ee; border-left:4px solid #9a83bf; border-radius:14px; padding:16px 19px; margin:12px 0 18px; }
.pv-guide-item { padding:12px 14px; border:1px solid #e9e3f1; border-radius:12px; background:rgba(255,255,255,.8); min-height:118px; }
[data-testid="stProgressBar"] > div > div { background-color:#8b70b5; }
/* Force a consistent light palette for Streamlit/BaseWeb controls and data grids. */
:root { color-scheme: light; }
html, body, [class*="css"], .stApp { color:#292637; }
.stMarkdown, .stMarkdown p, .stCaption, [data-testid="stCaptionContainer"], small { color:#514b60; }
[data-testid="stWidgetLabel"], [data-testid="stWidgetLabel"] p, label, legend { color:#393148 !important; font-size: .96rem !important; font-weight:600; }
[data-baseweb="select"] > div, [data-baseweb="select"] input, [data-baseweb="select"] span { background:#fff !important; color:#302a40 !important; }
[data-baseweb="popover"], [data-baseweb="menu"], [role="listbox"], [role="option"] { background:#fff !important; color:#302a40 !important; }
[data-baseweb="radio"] label, [data-testid="stRadio"] label, [data-testid="stSegmentedControl"] button { color:#302a40 !important; }
[data-testid="stDataFrame"] *, [data-testid="stTable"] * { color:#302a40; }
[data-testid="stDataFrame"] [role="columnheader"], [data-testid="stDataFrame"] [role="columnheader"] * { background:#eee8f7 !important; color:#302a40 !important; font-weight:700 !important; }
[data-testid="stExpander"] summary, [data-testid="stExpander"] summary p { color:#393148 !important; font-weight:650 !important; }
[data-testid="stFileUploader"] button { background:#7857a5 !important; color:#fff !important; border-color:#7857a5 !important; }
[data-testid="stFileUploader"] button * { color:#fff !important; }
[data-testid="stSegmentedControl"] button[aria-pressed="true"] { background:#7857a5 !important; color:#fff !important; border-color:#7857a5 !important; }
[data-testid="stSegmentedControl"] button[aria-pressed="true"] * { color:#fff !important; }
div.stButton > button[kind="primary"], div.stDownloadButton > button[kind="primary"] { background:#7857a5 !important; color:#fff !important; border-color:#7857a5 !important; }
div.stButton > button[kind="primary"] *, div.stDownloadButton > button[kind="primary"] * { color:#fff !important; }
div.stButton > button:focus, div.stDownloadButton > button:focus { outline:3px solid #c6b4e2 !important; outline-offset:2px; }
.stTextInput input::placeholder, .stTextArea textarea::placeholder { color:#898296 !important; opacity:1 !important; }
[data-testid="stAlert"] * { color:#302a40 !important; }
[data-testid="stAlert"] { border-radius:10px; }
.pv-muted { color:#595366 !important; font-size:.96rem; line-height:1.65; }
.pv-step-hint { color:#5f596d !important; font-size:.84rem; line-height:1.45; padding:0 .35rem .48rem 1.55rem; }
.pv-readiness { color:#393148 !important; font-size:.9rem; line-height:1.9; }
.pv-side-label { color:#655080 !important; font-size:.74rem; }
[data-testid="stSidebar"] .stCaption, [data-testid="stSidebar"] small { color:#595366 !important; font-size:.88rem !important; }
.pv-guide-item { min-height:130px; }
.pv-guide-item .pv-muted { margin-bottom:8px; }
[data-testid="stSidebar"] div.stButton > button[kind="primary"] { background:#7857a5 !important; color:#fff !important; border-color:#7857a5 !important; }
[data-testid="stSidebar"] div.stButton > button[kind="primary"] * { color:#fff !important; }
@media (max-width: 800px) { .pv-title {font-size:1.4rem;} .pv-guide-item {min-height:0;} }
</style>
''', unsafe_allow_html=True)

# Real plant rbcL accessions from a published Brassicaceae barcode dataset / GenBank.
# Demo sequences are fetched from NCBI at runtime rather than fabricated or hard-coded.
PLANT_DEMO_ACCESSIONS = [
    ('Arabidopsis_thaliana', 'AY174633.1'),
    ('Aethionema_grandiflora', 'AY167983.1'),
    ('Arabis_glabra', 'DQ310542.1'),
    ('Berteroa_incana', 'KM360667.1'),
    ('Anchonium_elichrysifolium', 'FN594834.1'),
]
# RefSeq mitochondrial genomes and annotated COI/COX1 start coordinates (1-based).
ANIMAL_COI_GENOMES = [
    ('Human_COI', 'NC_012920.1', 5904),
    ('Chimpanzee_COI', 'NC_001643.1', 5321),
    ('Gorilla_COI', 'NC_011120.1', 5326),
    ('Mouse_COI', 'NC_005089.1', 5328),
    ('Rat_COI', 'NC_001665.2', 5323),
]
DEMO_FASTA = ''
BASE_COLORS = {'A':'#b8d8f0','T':'#f5d58b','G':'#cbb8e8','C':'#a9d8c8','U':'#cbb8e8','-':'#d8d6df','N':'#eeeaf4'}
CREATOR = 'Anureet Dutta'
CO_CREATORS = ['Sreejeeta Kundu','Aishwarya Banik','Soham Pariyal','Ram Argha Nayak']

# Header with the supplied project logo.
h1, h2 = st.columns([1.5, 3.6], vertical_alignment='center')
with h1:
    if LOGO.exists(): st.image(str(LOGO), width=175)
    else: st.markdown('<div class="pv-title">🧬 PhyloVision</div>', unsafe_allow_html=True)
with h2:
    st.markdown('<div class="pv-kicker">From Sequences to Evolution</div><div style="font-size:1.28rem;font-weight:700;color:#393148;font-family:Manrope">Biological relationships, with room to breathe.</div><div class="pv-muted">A calm, precise workspace for sequence comparison and evolutionary exploration.</div>', unsafe_allow_html=True)
st.markdown('')

# Guided workflow and compact direct-access tool dock. Streamlit provides the native
# sidebar collapse/expand control, so this remains responsive without custom JS.
PAGE_GROUPS = {
    'Start here': ['Home', 'Sequence Input', 'Accession → FASTA'],
    'Analyze': ['Alignment Explorer', 'Phylogenetic Tree'],
    'Explore results': ['Sequence Similarity', 'Evolution Lens', 'Understand My Tree'],
    'Learn & about': ['Creators & About'],
}
ALL_PAGES = [p for group in PAGE_GROUPS.values() for p in group]
PAGE_HINTS = {
    'Home': 'Overview and guided start', 'Sequence Input': 'Paste, upload, or load demo FASTA',
    'Accession → FASTA': 'Fetch records from NCBI accession codes',
    'Alignment Explorer': 'Inspect aligned positions and differences',
    'Phylogenetic Tree': 'Build a tree with NJ or UPGMA',
    'Sequence Similarity': 'Compare pairwise sequence identity',
    'Evolution Lens': 'Explore evolutionary patterns', 'Understand My Tree': 'Interpret your tree in plain language',
    'Creators & About': 'Project credits and methodology',
}
# Initialize analysis state before drawing the sidebar so readiness is accurate on first load.
if 'fasta_text' not in st.session_state: st.session_state.fasta_text = ''
if 'tree' not in st.session_state: st.session_state.tree = None
if 'aligned_records' not in st.session_state: st.session_state.aligned_records = None
if 'dist_df' not in st.session_state: st.session_state.dist_df = None
if 'tree_method' not in st.session_state: st.session_state.tree_method = 'Neighbor Joining'
if 'sequence_type' not in st.session_state: st.session_state.sequence_type = 'DNA'
if 'bootstrap_support' not in st.session_state: st.session_state.bootstrap_support = {}
if 'tree_signature' not in st.session_state: st.session_state.tree_signature = None
if 'fetched_fasta_text' not in st.session_state: st.session_state.fetched_fasta_text = ''
if 'fetch_results' not in st.session_state: st.session_state.fetch_results = []
if 'page' not in st.session_state:
    requested_page = st.query_params.get('page', 'Home')
    st.session_state.page = requested_page if requested_page in ALL_PAGES else 'Home'
else:
    # Keep URL state in sync so browser back/forward can restore visited tools.
    url_page = st.query_params.get('page')
    if url_page in ALL_PAGES and url_page != st.session_state.page:
        st.session_state.page = url_page

def navigate_to(target):
    st.session_state.page = target
    try:
        st.query_params['page'] = target
    except Exception:
        pass

with st.sidebar:
    st.markdown('<div style="font-family:Manrope,sans-serif;font-size:1.28rem;font-weight:800;letter-spacing:-.035em">✳ PhyloVision</div><div style="color:#655080;font-size:.74rem;letter-spacing:.12em;font-weight:700;margin:.15rem 0 .65rem">BIOINFORMATICS WORKSPACE</div>', unsafe_allow_html=True)
    has_input = bool(st.session_state.get('fasta_text', '').strip())
    try:
        progress_records = list(SeqIO.parse(io.StringIO(st.session_state.get('fasta_text', '')), 'fasta'))
        has_alignment = len(progress_records) >= 2 and len({len(r.seq) for r in progress_records}) == 1
    except Exception:
        has_alignment = False
    sig = st.session_state.get('tree_signature')
    has_tree = st.session_state.get('tree') is not None and bool(sig) and sig[0] == st.session_state.get('fasta_text', '') and sig[1] == st.session_state.get('sequence_type', 'DNA')
    done = sum([has_input, has_alignment, has_tree])
    st.markdown('<div class="pv-side-label">PROGRESS</div>', unsafe_allow_html=True)
    st.progress(done / 3, text=f'{done}/3 core milestones')
    st.markdown(f'<div class="pv-readiness">{"✓" if has_input else "–"} Sequence input<br>{"✓" if has_alignment else "–"} Alignment available<br>{"✓" if has_tree else "–"} Tree built</div><div class="pv-step-hint">Workflow status only; not a certificate of biological validity.</div><div class="pv-side-divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="pv-side-label">QUICK ACTION SEARCH</div>', unsafe_allow_html=True)
    search_query = st.text_input('Search tools', placeholder='Find a tool or action…', label_visibility='collapsed', key='quick_action_search')
    if search_query.strip():
        matches = [name for name in ALL_PAGES if search_query.lower() in name.lower() or search_query.lower() in PAGE_HINTS[name].lower()]
        if matches:
            chosen_match = st.selectbox('Matching actions', matches, label_visibility='collapsed', key='quick_action_match')
            if st.button('Open selected tool →', use_container_width=True, key='open_quick_action'):
                navigate_to(chosen_match)
                st.rerun()
        else:
            st.caption('No matching tool. Try “tree”, “FASTA”, “similarity”, or “alignment”.')
    st.markdown('<div class="pv-side-divider"></div><div class="pv-side-label">GUIDED LAB</div><div class="pv-muted" style="font-size:.82rem;margin:.25rem 0 .65rem">Follow the steps in order, or jump ahead whenever you are ready.</div>', unsafe_allow_html=True)
    guided_steps = [
        ('1 · Prepare sequences', 'Sequence Input', 'Paste, upload, or load demo data'),
        ('2 · Fetch by accession (Optional)', 'Accession → FASTA', 'Optional: retrieve records from NCBI'),
        ('3 · Inspect alignment', 'Alignment Explorer', 'Check lengths, gaps, and positions'),
        ('4 · Build a tree', 'Phylogenetic Tree', 'Infer relationships from aligned data'),
        ('5 · Explore findings', 'Sequence Similarity', 'Compare and interpret results'),
    ]
    for label, target, hint in guided_steps:
        active = st.session_state.page == target
        completed = {
            'Sequence Input': has_input,
            'Accession → FASTA': bool(st.session_state.get('fetched_fasta_text')),
            'Alignment Explorer': has_alignment,
            'Phylogenetic Tree': has_tree,
            'Sequence Similarity': has_alignment,
        }.get(target, False)
        prefix = '✓  ' if completed else ('›  ' if active else '   ')
        if st.button(prefix + label, key='guided_'+target, use_container_width=True, type='primary' if active else 'secondary'):
            navigate_to(target)
            st.rerun()
        st.markdown(f'<div class="pv-step-hint">{hint}</div>', unsafe_allow_html=True)
    st.markdown('<div class="pv-side-divider"></div><div class="pv-side-label">TOOL DOCK · DIRECT ACCESS</div>', unsafe_allow_html=True)
    dock_pages = ['Home', 'Sequence Input', 'Accession → FASTA', 'Alignment Explorer', 'Phylogenetic Tree', 'Sequence Similarity', 'Evolution Lens', 'Understand My Tree', 'Creators & About']
    dock_choice = st.selectbox('Open a tool directly', dock_pages, index=dock_pages.index(st.session_state.page) if st.session_state.page in dock_pages else 0, label_visibility='collapsed', key='tool_dock_select')
    if st.button('Go to tool ↗', use_container_width=True, key='tool_dock_go'):
        navigate_to(dock_choice)
        st.rerun()

    st.markdown('<div class="pv-side-divider"></div><div class="pv-muted" style="font-size:.88rem">Educational analysis · Interpret results in light of sequence quality and method.</div>', unsafe_allow_html=True)

page = st.session_state.page

if 'fasta_text' not in st.session_state: st.session_state.fasta_text = ''
if 'tree' not in st.session_state: st.session_state.tree = None
if 'aligned_records' not in st.session_state: st.session_state.aligned_records = None
if 'dist_df' not in st.session_state: st.session_state.dist_df = None
if 'tree_method' not in st.session_state: st.session_state.tree_method = 'Neighbor Joining'
if 'sequence_type' not in st.session_state: st.session_state.sequence_type = 'DNA'
if 'bootstrap_support' not in st.session_state: st.session_state.bootstrap_support = {}
if 'tree_signature' not in st.session_state: st.session_state.tree_signature = None
if 'fetched_fasta_text' not in st.session_state: st.session_state.fetched_fasta_text = ''
if 'fetch_results' not in st.session_state: st.session_state.fetch_results = []


def parse_fasta(text):
    if not text.strip(): raise ValueError('Please paste or upload FASTA sequences first.')
    try: records = list(SeqIO.parse(io.StringIO(text), 'fasta'))
    except Exception as e: raise ValueError(f'Could not read FASTA: {e}')
    if len(records) < 2: raise ValueError('Please provide at least 2 FASTA sequences to compare.')
    seen=set()
    for i,r in enumerate(records):
        r.id = re.sub(r'\s+','_',r.id.strip()) or f'Sequence_{i+1}'
        r.name=r.id; r.description=''
        if r.id in seen: raise ValueError(f'Duplicate sequence name: {r.id}. Please use unique FASTA headers.')
        seen.add(r.id)
        r.seq = Seq(str(r.seq).upper().replace(' ','').replace('.', '-'))
        if not str(r.seq): raise ValueError(f'{r.id} has an empty sequence.')
    return records


def validate_records(records, seq_type):
    allowed = set('ACGTUNRYKMSWBDHV-') if seq_type == 'DNA' else set('ABCDEFGHIKLMNPQRSTVWXYZJUO*-')
    issues=[]
    for r in records:
        invalid=sorted(set(str(r.seq).upper())-allowed)
        if invalid: issues.append(f'{r.id}: unsupported character(s) {", ".join(invalid)}')
    lengths={len(r.seq) for r in records}
    if len(lengths)>1: issues.append('Sequences have different lengths. They may not be aligned yet; this version can compare them only after equal-length alignment.')
    if issues: raise ValueError('\n'.join(issues))
    return records


def comparable_pair(a, b, sequence_type='DNA'):
    # Exclude gaps and ambiguity codes from BOTH sides. Valid protein residues are retained.
    excluded = set('-?.')
    if sequence_type == 'DNA':
        excluded.update('NRYKMSWBDHVX')
    else:
        excluded.update('XBZJ*')
    pairs = [(x, y) for x, y in zip(str(a).upper(), str(b).upper())
             if x not in excluded and y not in excluded]
    compared = len(pairs)
    matches = sum(x == y for x, y in pairs)
    return matches, compared


def pairwise_p_distance(a, b, sequence_type='DNA'):
    matches, compared = comparable_pair(a, b, sequence_type)
    if compared == 0: return np.nan
    return (compared - matches) / compared


def make_similarity_matrix(records, sequence_type='DNA'):
    """Pairwise percent identity; ambiguous/missing positions are excluded pairwise."""
    names = [r.id for r in records]
    mat = np.full((len(records), len(records)), np.nan, dtype=float)
    for i, record in enumerate(records):
        _, compared = comparable_pair(record.seq, record.seq, sequence_type)
        if compared: mat[i, i] = 100.0
        for j in range(i + 1, len(records)):
            matches, compared = comparable_pair(record.seq, records[j].seq, sequence_type)
            value = (100.0 * matches / compared) if compared else np.nan
            mat[i, j] = mat[j, i] = value
    return pd.DataFrame(mat, index=names, columns=names)


def make_distance_matrix(records, sequence_type='DNA'):
    names = [r.id for r in records]
    mat = np.zeros((len(records), len(records)))
    for i in range(len(records)):
        for j in range(i + 1, len(records)):
            d = pairwise_p_distance(records[i].seq, records[j].seq, sequence_type)
            mat[i, j] = mat[j, i] = d if np.isfinite(d) else 1.0
    return pd.DataFrame(mat, index=names, columns=names)


def build_tree(records, model='Neighbor Joining', sequence_type='DNA'):
    """Build a distance tree from an explicit Biopython MultipleSeqAlignment."""
    alignment = MultipleSeqAlignment([r[:] for r in records])
    skip_letters = '-?.NRYKMSWBDHVX' if sequence_type == 'DNA' else '-?.XBZJ*'
    calculator = DistanceCalculator('identity', skip_letters=skip_letters)
    distance_matrix = calculator.get_distance(alignment)
    constructor = DistanceTreeConstructor()
    if model == 'UPGMA':
        tree = constructor.upgma(distance_matrix)
        tree.rooted = True
    else:
        tree = constructor.nj(distance_matrix)
        tree.rooted = False
    return tree


def _split_key(names):
    """Canonical key for an unrooted bipartition."""
    all_names = frozenset(names)
    # A stable lexical representation makes equivalent splits comparable.
    return tuple(sorted(all_names))


def _tree_splits(tree):
    """Return non-trivial unrooted splits for approximate bootstrap support."""
    all_taxa = frozenset(t.name for t in tree.get_terminals())
    splits = set()
    for clade in tree.get_nonterminals():
        side = frozenset(t.name for t in clade.get_terminals())
        other = all_taxa - side
        if len(side) < 2 or len(other) < 2:
            continue
        # For an unrooted split, the smaller side is canonical; ties are lexical.
        a, b = tuple(sorted(side)), tuple(sorted(other))
        splits.add(min(a, b) if len(a) == len(b) else (a if len(a) < len(b) else b))
    return splits


def bootstrap_tree(records, model='Neighbor Joining', replicates=100, seed=17, sequence_type='DNA'):
    """Column-resampling bootstrap; returns the original tree and split support %.

    This is a standard non-parametric site bootstrap for an already-aligned
    dataset. It is intentionally capped by the UI for a student-hosted app.
    """
    base_tree = build_tree(records, model, sequence_type)
    n_sites = len(records[0].seq)
    if n_sites < 2:
        raise ValueError('At least two alignment positions are needed for bootstrap resampling.')
    rng = np.random.default_rng(seed)
    base_splits = _tree_splits(base_tree)
    counts = {split: 0 for split in base_splits}
    alignment_rows = [str(r.seq) for r in records]
    for _ in range(int(replicates)):
        cols = rng.integers(0, n_sites, size=n_sites)
        sampled = [SeqRecord(Seq(''.join(row[i] for i in cols)), id=r.id, description='')
                   for r, row in zip(records, alignment_rows)]
        replicate_tree = build_tree(sampled, model, sequence_type)
        rep_splits = _tree_splits(replicate_tree)
        for split in base_splits:
            if split in rep_splits:
                counts[split] += 1
    support = {split: 100.0 * count / int(replicates) for split, count in counts.items()}
    return base_tree, support


def pdf_report(records, tree, model, support, sequence_type):
    """Create a compact, data-grounded PDF analysis report in memory."""
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=0.65*inch, leftMargin=0.65*inch,
                            topMargin=0.6*inch, bottomMargin=0.6*inch)
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name='PVTitle', parent=styles['Title'], textColor=colors.HexColor('#55328a')))
    story = [Paragraph('PhyloVision Analysis Report', styles['PVTitle']),
             Paragraph('From Sequences to Evolution · Educational analysis', styles['Normal']), Spacer(1, 12),
             Paragraph('Analysis summary', styles['Heading2']),
             Paragraph(f'Sequence type: {sequence_type} · Sequences: {len(records)} · Alignment length: {len(records[0].seq)} positions · Method: {model} · Distance: uncorrected p-distance (identity)', styles['BodyText']), Spacer(1, 8),
             Paragraph('Input sequences', styles['Heading2'])]
    data = [['Sequence name', 'Length', 'Ambiguous/gap positions']]
    for r in records:
        seq = str(r.seq).upper()
        data.append([r.id, str(len(seq)), str(sum(ch in '-N?X' for ch in seq))])
    table = Table(data, repeatRows=1, colWidths=[3.2*inch, 1.1*inch, 2.0*inch])
    table.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#ded4f2')),('TEXTCOLOR',(0,0),(-1,0),colors.HexColor('#2b2142')),('GRID',(0,0),(-1,-1),0.4,colors.HexColor('#c8c3d5')),('PADDING',(0,0),(-1,-1),6),('VALIGN',(0,0),(-1,-1),'TOP')]))
    story += [table, Spacer(1, 12), Paragraph('Closest pair in inferred tree', styles['Heading2'])]
    terms = tree.get_terminals()
    pairs = sorted((tree.distance(a,b), a.name, b.name) for i,a in enumerate(terms) for b in terms[i+1:])
    if pairs:
        d, a, b = pairs[0]
        story.append(Paragraph(f'{a} and {b} have the smallest tree-path distance ({d:.5f}). This does not imply direct ancestry.', styles['BodyText']))
    story += [Spacer(1, 8), Paragraph('Bootstrap support', styles['Heading2'])]
    if support:
        rows = [['Split (taxa on smaller side)', 'Support']]
        rows += [[', '.join(k), f'{v:.1f}%'] for k,v in sorted(support.items(), key=lambda item: item[1], reverse=True)]
        stbl = Table(rows, repeatRows=1, colWidths=[5.5*inch, 1.0*inch])
        stbl.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#ded4f2')),('GRID',(0,0),(-1,-1),0.4,colors.HexColor('#c8c3d5')),('PADDING',(0,0),(-1,-1),5)]))
        story.append(stbl)
    else:
        story.append(Paragraph('Bootstrap was not run for this analysis.', styles['BodyText']))
    story += [Spacer(1, 12), Paragraph('Method and limitations', styles['Heading2']),
              Paragraph('The input sequences were treated as an existing multiple sequence alignment. Neighbor Joining or UPGMA was applied to an identity-based distance matrix. Bootstrap percentages, when shown, come from resampling alignment columns with replacement and rebuilding trees; they measure split recovery under this procedure, not the probability that a clade is true. The app does not perform sequence alignment automatically. Confirm sequence homology, alignment quality, data source and model suitability before drawing biological conclusions.', styles['BodyText'])]
    doc.build(story)
    return buffer.getvalue()


def rectangular_tree_figure(tree):
    # Render branch lengths with an elegant rectangular phylogram.
    terminals=tree.get_terminals(); n=len(terminals)
    ymap={term: i for i,term in enumerate(terminals)}
    def assign_y(clade):
        if clade.is_terminal(): return ymap[clade]
        vals=[assign_y(c) for c in clade.clades]
        ymap[clade]=sum(vals)/len(vals)
        return ymap[clade]
    assign_y(tree.root)
    xmap={tree.root:0.0}
    def assign_x(clade, parent_x):
        for c in clade.clades:
            length=float(c.branch_length or 0.01)
            xmap[c]=parent_x+max(length,0.002)
            assign_x(c,xmap[c])
    assign_x(tree.root,0.0)
    fig=go.Figure()
    palette=['#d58be8','#68c9c3','#f1bd65','#77a7f8','#e78a9e','#93d59a']
    for idx,clade in enumerate(tree.get_nonterminals(order='preorder')):
        if clade is tree.root: continue
    for parent in tree.find_clades(order='preorder'):
        if parent.is_terminal(): continue
        children=parent.clades
        if not children: continue
        ys=[ymap[c] for c in children]
        fig.add_trace(go.Scatter(x=[xmap[parent],xmap[parent]],y=[min(ys),max(ys)],mode='lines',line=dict(color='#8b70b5',width=2.2),hoverinfo='skip',showlegend=False))
        for c in children:
            color=palette[terminals.index(c)%len(palette)] if c.is_terminal() else '#8b83c8'
            fig.add_trace(go.Scatter(x=[xmap[parent],xmap[c]],y=[ymap[c],ymap[c]],mode='lines',line=dict(color='#8b70b5',width=2.2),hoverinfo='skip',showlegend=False))
    maxx=max(xmap.values()) if xmap else 1
    for t in terminals:
        fig.add_trace(go.Scatter(x=[xmap[t]],y=[ymap[t]],mode='markers+text',text=[t.name],textposition='middle right',textfont=dict(color='#393448',size=11),marker=dict(size=7,color=palette[terminals.index(t)%len(palette)]),hovertemplate=f'<b>{t.name}</b><br>Root distance: {xmap[t]:.4f}<extra></extra>',showlegend=False))
    fig.update_layout(height=max(430,n*42),margin=dict(l=18,r=210,t=18,b=18),paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)',template='plotly_white',xaxis=dict(title='Accumulated branch length (relative evolutionary change)',showgrid=True,gridcolor='rgba(130,145,190,.15)',zeroline=False,color='#514a60',rangemode='tozero',range=[0,maxx*1.65 if maxx else 1]),yaxis=dict(visible=False,autorange='reversed'),font=dict(family='DM Sans',color='#393448'),showlegend=False)
    return fig



def radial_tree_figure(tree):
    """Radial phylogram layout for a less conventional, botanical-inspired tree view."""
    terminals = tree.get_terminals()
    n = len(terminals)
    if n < 2:
        return rectangular_tree_figure(tree)
    angle_map = {term: (2 * math.pi * i / n) for i, term in enumerate(terminals)}
    def assign_angle(clade):
        if clade.is_terminal(): return angle_map[clade]
        vals = [assign_angle(child) for child in clade.clades]
        angle = math.atan2(sum(math.sin(v) for v in vals), sum(math.cos(v) for v in vals))
        angle_map[clade] = angle
        return angle
    assign_angle(tree.root)
    radius_map = {tree.root: 0.0}
    def assign_radius(clade, parent_r):
        for child in clade.clades:
            r = parent_r + max(float(child.branch_length or 0.0), 0.008)
            radius_map[child] = r
            assign_radius(child, r)
    assign_radius(tree.root, 0.0)
    max_r = max(radius_map.values()) or 1.0
    palette = ['#b9d0a8','#c8b5e8','#d6c69a','#92c7b7','#d7a7b9','#a5b9dc']
    fig = go.Figure()
    def xy(clade):
        a, r = angle_map[clade], radius_map[clade]
        return r * math.cos(a), r * math.sin(a)
    for parent in tree.find_clades(order='preorder'):
        if parent.is_terminal(): continue
        px, py = xy(parent)
        for child in parent.clades:
            cx, cy = xy(child)
            color = palette[terminals.index(child) % len(palette)] if child.is_terminal() else '#91aa91'
            fig.add_trace(go.Scatter(x=[px,cx], y=[py,cy], mode='lines', line=dict(color='#8b70b5',width=2.2), hoverinfo='skip', showlegend=False))
    for term in terminals:
        x, y = xy(term); a = angle_map[term]
        label_radius = radius_map[term] + max_r * .07
        lx, ly = label_radius * math.cos(a), label_radius * math.sin(a)
        fig.add_trace(go.Scatter(x=[x],y=[y],mode='markers',marker=dict(size=7,color=palette[terminals.index(term)%len(palette)],line=dict(color='#e6eddc',width=.8)),name=term.name,hovertemplate=f'<b>{term.name}</b><br>Root distance: {radius_map[term]:.4f}<extra></extra>',showlegend=False))
        fig.add_annotation(x=lx,y=ly,text=term.name,showarrow=False,font=dict(size=10,color='#393448'),xanchor='left' if math.cos(a)>=0 else 'right',yanchor='middle')
    limit = max_r * 1.28
    fig.update_layout(height=max(520,n*34),margin=dict(l=55,r=170,t=35,b=35),paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)',font=dict(family='DM Sans',color='#393448'),template='plotly_white',showlegend=False,xaxis=dict(visible=False,range=[-limit,limit],scaleanchor='y',scaleratio=1),yaxis=dict(visible=False,range=[-limit,limit]),hovermode='closest')
    return fig


def alignment_html(records, start=0, max_cols=60):
    full_len=max(len(r.seq) for r in records)
    max_len=min(max(0,full_len-start),max_cols)
    html='<div style="overflow-x:auto;max-width:100%"><table style="border-collapse:separate;border-spacing:2px;font:13px monospace;color:#302a40;width:max-content"><thead><tr><th style="text-align:left;position:sticky;left:0;background:#eee8f7;color:#302a40;padding:6px">Sequence</th>'
    for i in range(max_len):
        if (start+i) % 10 == 0: html+=f'<th colspan="1" style="color:#514a60;padding:2px">{start+i+1}</th>'
        else: html+='<th style="padding:2px"></th>'
    html+='</tr></thead><tbody>'
    for r in records:
        html+=f'<tr><td style="padding:4px 8px;position:sticky;left:0;background:#f1eef8;color:#302a40;white-space:nowrap">{r.id}</td>'
        for ch in str(r.seq)[start:start+max_len]:
            c=BASE_COLORS.get(ch,'#8b8fa7')
            html+=f'<td style="background:{c};color:#10162b;border-radius:3px;text-align:center;min-width:17px;padding:3px">{ch}</td>'
        html+='</tr>'
    # Derived consensus and conservation rows; ambiguous symbols and gaps are not counted as bases.
    rows = [str(r.seq).upper() for r in records]
    consensus=[]; markers=[]
    for j in range(start, start+max_len):
        chars=[row[j] for row in rows if row[j] not in '-.?NX']
        if not chars:
            consensus.append('·'); markers.append(' ')
        else:
            counts={c:chars.count(c) for c in set(chars)}
            top=max(counts.values()); winners=sorted(c for c,nc in counts.items() if nc==top)
            consensus.append(winners[0] if len(winners)==1 else 'N')
            markers.append('*' if len(winners)==1 and top==len(chars) else (':' if top/len(chars)>=0.75 else '.'))
    html+='<tr><td style="padding:4px 8px;position:sticky;left:0;background:#eee8f7;color:#302a40;font-weight:700;white-space:nowrap">Conservation</td>'
    for mark in markers: html+=f'<td style="background:#eee8f7;color:#514a60;text-align:center;min-width:17px;padding:3px">{mark}</td>'
    html+='</tr><tr><td style="padding:4px 8px;position:sticky;left:0;background:#e5dcf2;color:#302a40;font-weight:700;white-space:nowrap">Consensus</td>'
    for ch in consensus: html+=f'<td style="background:#e5dcf2;color:#302a40;text-align:center;min-width:17px;padding:3px">{ch}</td>'
    html+='</tr></tbody></table></div>'
    return html, max_len


def explain_tree(tree, names):
    terms=tree.get_terminals()
    if len(terms)<2: return ['Not enough sequences to explain the tree.']
    pairs=[]
    for i,a in enumerate(terms):
        for b in terms[i+1:]:
            try: d=tree.distance(a,b)
            except Exception: continue
            pairs.append((d,a.name,b.name))
    pairs.sort()
    findings=[]
    if pairs:
        d,a,b=pairs[0]
        tied=[(x,y) for dist,x,y in pairs if math.isclose(dist,d,rel_tol=1e-7,abs_tol=1e-9)]
        pair_text='; '.join(f'{x} and {y}' for x,y in tied[:4])
        tie_note=f' ({len(tied)} pairs tie)' if len(tied)>1 else ''
        findings.append(f'**Closest pair in this tree:** {pair_text} have the smallest patristic distance ({d:.4f}){tie_note}. This is a result of the supplied sequences and method, not proof that one modern organism descended from the other.')
    root=tree.root
    if root.clades:
        largest=max(root.clades,key=lambda c:len(c.get_terminals()))
        if len(largest.get_terminals())>=2:
            members=', '.join(t.name for t in largest.get_terminals())
            findings.append(f'**One large grouping:** {members} appear together on one side of the current unrooted tree. Tree orientation/rooting affects how the grouping is displayed.')
    findings.append('**Branch lengths:** longer paths represent greater estimated sequence change under the distance calculation. They do not automatically represent elapsed time.')
    findings.append('**Support:** branch confidence is only available when bootstrap resampling is run; branch lengths and visual spacing are not confidence values.')
    return findings


def align_reference_region(raw_records, region_len=450):
    # Star-align homologous marker sequences to the first record using a global pairwise
    # aligner. This is deliberately labelled as a teaching-grade alignment, not a research MSA.
    from Bio.Align import PairwiseAligner
    reference = str(raw_records[0].seq)
    region_len = min(int(region_len), len(reference))
    aligner = PairwiseAligner()
    aligner.mode = 'global'; aligner.match_score = 2.0; aligner.mismatch_score = -1.0
    aligner.open_gap_score = -5.0; aligner.extend_gap_score = -0.5
    mappings=[]; insertions=[]
    for rec in raw_records:
        query=str(rec.seq); base_map=['-']*len(reference); ins_map={i:'' for i in range(len(reference)+1)}
        if rec.id == raw_records[0].id:
            base_map=list(reference)
        else:
            coords=aligner.align(reference,query)[0].coordinates
            for k in range(coords.shape[1]-1):
                r0,r1=int(coords[0,k]),int(coords[0,k+1]); q0,q1=int(coords[1,k]),int(coords[1,k+1])
                dr,dq=r1-r0,q1-q0
                if dr==0 and dq>0:
                    ins_map[r0]=ins_map.get(r0,'')+query[q0:q1]
                elif dr>0 and dq>0:
                    for off in range(min(dr,dq)):
                        if r0+off<len(base_map) and q0+off<len(query): base_map[r0+off]=query[q0+off]
        mappings.append(base_map); insertions.append(ins_map)
    widths={b:max(len(ins.get(b,'')) for ins in insertions) for b in range(region_len+1)}
    aligned=[]
    for rec,base_map,ins_map in zip(raw_records,mappings,insertions):
        pieces=[]
        for pos in range(region_len):
            ins=ins_map.get(pos,'')[:widths[pos]]; pieces.extend([ins+'-'*(widths[pos]-len(ins)),base_map[pos]])
        ins=ins_map.get(region_len,'')[:widths[region_len]]; pieces.append(ins+'-'*(widths[region_len]-len(ins)))
        aligned.append(SeqRecord(Seq(''.join(pieces)),id=rec.id,description=rec.description))
    validate_records(aligned,'DNA')
    return aligned


def store_teaching_dataset(aligned, source):
    out=io.StringIO(); SeqIO.write(aligned,out,'fasta')
    st.session_state.fasta_text=out.getvalue(); st.session_state.sequence_type='DNA'
    st.session_state.tree=None; st.session_state.tree_signature=None; st.session_state.aligned_records=None
    st.session_state.dist_df=None; st.session_state.bootstrap_support={}
    st.session_state.demo_source=source; st.session_state.demo_fasta_snapshot=out.getvalue()


def fetch_plant_teaching_example(email=''):
    Entrez.email=email.strip() or None
    raw_records=[]
    for display_name,accession in PLANT_DEMO_ACCESSIONS:
        with Entrez.efetch(db='nuccore',id=accession,rettype='fasta',retmode='text') as handle:
            recs=list(SeqIO.parse(handle,'fasta'))
        if not recs: raise ValueError(f'NCBI returned no sequence for {accession}.')
        sequence=re.sub(r'[^ACGTNRYKMSWBDHV]', '', str(recs[0].seq).upper().replace('U','T'))
        if len(sequence)<450: raise ValueError(f'{accession} returned only {len(sequence)} usable bases; expected at least 450.')
        raw_records.append(SeqRecord(Seq(sequence),id=display_name,description=f'rbcL | GenBank:{accession}'))
    aligned=align_reference_region(raw_records,450)
    store_teaching_dataset(aligned,'Plant rbcL · NCBI GenBank: '+', '.join(a for _,a in PLANT_DEMO_ACCESSIONS))


def fetch_animal_coi_example(email=''):
    Entrez.email=email.strip() or None
    raw_records=[]
    for display_name,accession,start_1based in ANIMAL_COI_GENOMES:
        with Entrez.efetch(db='nuccore',id=accession,rettype='fasta',retmode='text') as handle:
            recs=list(SeqIO.parse(handle,'fasta'))
        if not recs: raise ValueError(f'NCBI returned no mitochondrial genome for {accession}.')
        genome=str(recs[0].seq).upper().replace('U','T')
        fragment=genome[start_1based-1:start_1based-1+450]
        if len(fragment)<450: raise ValueError(f'{accession} did not contain the expected COI interval starting at {start_1based}.')
        raw_records.append(SeqRecord(Seq(fragment),id=display_name,description=f'COI 450-nt fragment | GenBank:{accession} coordinates:{start_1based}-{start_1based+449}'))
    aligned=align_reference_region(raw_records,450)
    store_teaching_dataset(aligned,'Animal COI · NCBI RefSeq mitochondrial genomes: '+', '.join(a for _,a,_ in ANIMAL_COI_GENOMES))


def current_records():
    records=parse_fasta(st.session_state.fasta_text)
    validate_records(records, st.session_state.sequence_type)
    return records

# Home
if page == 'Home':
    st.markdown('<div class="pv-card"><div class="pv-kicker">Welcome to your sequence workspace</div><div class="pv-title">From raw sequences to clearer relationships.</div><div class="pv-muted">PhyloVision brings sequence preparation, alignment inspection, similarity comparison and phylogenetic interpretation into one calm, student-friendly workflow.</div></div>',unsafe_allow_html=True)
    st.markdown('<div class="pv-accession-feature"><div class="pv-kicker">Start with real data</div><div class="pv-section" style="margin:.3rem 0">Load a real plant rbcL or mammal COI dataset</div><div class="pv-muted">Choose between five real plant rbcL records and five mammalian mitochondrial COI fragments (human, chimpanzee, gorilla, mouse, rat). Records are fetched from NCBI, aligned to a common reference region for teaching, and loaded into the analysis workflow. Internet access is required.</div></div>', unsafe_allow_html=True)
    demo_kind = st.radio('Teaching dataset', ['Plant rbcL · Brassicaceae', 'Animal COI · mammals'], horizontal=True, key='home_demo_kind')
    cta1, cta2 = st.columns([1.2,1])
    with cta1:
        if st.button('Load demo data → Start', key='home_load_demo', type='primary', use_container_width=True):
            try:
                with st.spinner('Fetching real gene records from NCBI and aligning the teaching region…'):
                    if demo_kind.startswith('Plant'):
                        fetch_plant_teaching_example(st.session_state.get('ncbi_email',''))
                    else:
                        fetch_animal_coi_example(st.session_state.get('ncbi_email',''))
                navigate_to('Sequence Input'); st.rerun()
            except Exception as exc:
                st.error(f'Could not load the NCBI teaching dataset: {exc}. Check your connection or use Accession → FASTA to try again.')
    with cta2:
        if st.button('Fetch by accession →', key='home_accession_cta', use_container_width=True):
            navigate_to('Accession → FASTA'); st.rerun()
    st.markdown('<div class="pv-section">Quick guide to PhyloVision</div><div class="pv-muted" style="margin-bottom:12px">Use the numbered Guided Lab steps in the sidebar if you are new. For direct access, use Tool Dock; to find something quickly, use Quick Action Search.</div>', unsafe_allow_html=True)
    guide = [
        ('01 · Sequence Input','Paste or upload FASTA, or load demo sequences.','Sidebar → Guided Lab → Prepare sequences'),
        ('02 · Accession → FASTA','Fetch records from NCBI using accession codes; preview and download FASTA.','Sidebar → Guided Lab → Fetch by accession'),
        ('03 · Alignment Explorer','Inspect aligned positions, gaps and differences. Sequences must be aligned before tree building.','Sidebar → Guided Lab → Inspect alignment'),
        ('04 · Phylogenetic Tree','Build a distance-based tree using available methods and explore branching patterns.','Sidebar → Guided Lab → Build a tree'),
        ('05 · Sequence Similarity','Compare pairwise identity across comparable sequence positions.','Tool Dock → Sequence Similarity'),
        ('06 · Evolution Lens','Explore patterns and summaries from the current analysis.','Tool Dock → Evolution Lens'),
        ('07 · Understand My Tree','Read a plain-language explanation of tree structure and its limitations.','Tool Dock → Understand My Tree'),
        ('08 · Creators & About','Find project credits, methodology and current scientific limitations.','Tool Dock → Creators & About'),
    ]
    for offset in range(0, len(guide), 2):
        cols = st.columns(2)
        for col, item in zip(cols, guide[offset:offset+2]):
            with col:
                st.markdown(f'<div class="pv-guide-item"><div class="pv-section">{item[0]}</div><div class="pv-muted">{item[1]}</div></div>', unsafe_allow_html=True)
                target = {'01 · Sequence Input':'Sequence Input','02 · Accession → FASTA':'Accession → FASTA','03 · Alignment Explorer':'Alignment Explorer','04 · Phylogenetic Tree':'Phylogenetic Tree','05 · Sequence Similarity':'Sequence Similarity','06 · Evolution Lens':'Evolution Lens','07 · Understand My Tree':'Understand My Tree','08 · Creators & About':'Creators & About'}[item[0]]
                if st.button('Open ' + target + ' →', key='home_guide_'+target, use_container_width=True):
                    navigate_to(target); st.rerun()
    st.markdown('<div class="pv-card" style="margin-top:14px"><div class="pv-section">Suggested beginner workflow</div><ol><li>Prepare sequences or fetch them using accession codes.</li><li>Inspect the alignment and confirm sequences are suitable for comparison.</li><li>Build a phylogenetic tree.</li><li>Explore similarity and interpret the result cautiously.</li></ol><div class="pv-muted">The sidebar readiness indicator reflects session state, not scientific validity. The plant teaching dataset is fetched from traceable GenBank accessions when selected; equal lengths do not prove alignment.</div></div>',unsafe_allow_html=True)

elif page == 'Accession → FASTA':
    st.markdown('<div class="pv-kicker">Sequence acquisition</div><div class="pv-title">Accession → FASTA</div><div class="pv-muted">Fetch sequences from NCBI by accession number, preview the returned FASTA, and bring records into your existing workflow.</div>', unsafe_allow_html=True)
    st.markdown('<div class="pv-card"><div class="pv-section">Fetch records from NCBI</div><div class="pv-muted">Enter one accession per line. Choose the database explicitly; an accession by itself does not always tell us whether it represents DNA or protein.</div></div>', unsafe_allow_html=True)
    db_label = st.radio('NCBI database', ['Nucleotide (DNA/RNA)', 'Protein'], horizontal=True, key='accession_database')
    db_name = 'nucleotide' if db_label.startswith('Nucleotide') else 'protein'
    fetched_type = 'DNA' if db_name == 'nucleotide' else 'Protein'
    if st.button('Try an example accession', key='try_accession_example'):
        st.session_state['accession_codes_input'] = 'NP_008188.1' if db_label == 'Protein' else 'AY174633.1'
        st.rerun()
    accession_text = st.text_area('Accession codes', placeholder='Paste accession codes here, one per line…', height=115, key='accession_codes_input', help='Paste accession codes, one per line. Commas and semicolons are also accepted.')
    contact_email = st.text_input('Contact email (recommended by NCBI)', placeholder='you@example.com', key='ncbi_email', help='Kept in this active Streamlit session so you do not need to retype it on every visit. It is not written to a project file.')
    max_accessions = 20
    accessions = [x.strip() for x in re.split(r'[\n,;\s]+', accession_text) if x.strip()]
    # Preserve order while removing duplicate accessions.
    accessions = list(dict.fromkeys(accessions))
    if accessions:
        st.caption(f'{len(accessions)} unique accession(s) entered · limit {max_accessions} per request')
    fetch_col, clear_col = st.columns([1, 1])
    with fetch_col:
        fetch_clicked = st.button('Fetch FASTA records', type='primary', use_container_width=True, disabled=not accessions or len(accessions) > max_accessions)
    with clear_col:
        if st.button('Clear fetched results', use_container_width=True):
            st.session_state.fetched_fasta_text = ''
            st.session_state.fetch_results = []
            st.rerun()
    if len(accessions) > max_accessions:
        st.warning(f'Please fetch no more than {max_accessions} accessions at a time to keep requests manageable.')
    if fetch_clicked:
        Entrez.email = contact_email.strip() or None
        fetched_records = []
        statuses = []
        progress = st.progress(0, text='Connecting to NCBI…')
        for idx, accession in enumerate(accessions):
            try:
                with Entrez.efetch(db=db_name, id=accession, rettype='fasta', retmode='text') as handle:
                    returned = list(SeqIO.parse(handle, 'fasta'))
                if not returned:
                    statuses.append({'Accession': accession, 'Status': 'No FASTA record returned', 'Details': 'Check the accession and selected database.'})
                else:
                    for record in returned:
                        record.id = re.sub(r'\s+', '_', record.id.strip())
                        record.name = record.id
                        record.description = (record.description or record.id).strip()
                        record.seq = Seq(str(record.seq).upper().replace(' ', ''))
                        fetched_records.append(record)
                    statuses.append({'Accession': accession, 'Status': 'Fetched', 'Details': f'{len(returned)} record(s) returned'})
            except Exception as exc:
                message = str(exc).strip().replace('\n', ' ')
                if len(message) > 220: message = message[:217] + '...'
                statuses.append({'Accession': accession, 'Status': 'Failed', 'Details': message or 'Network or NCBI request failed.'})
            progress.progress((idx + 1) / len(accessions), text=f'Processed {idx + 1} of {len(accessions)} accession(s)')
        progress.empty()
        if fetched_records:
            # Avoid duplicate FASTA IDs, because downstream PhyloVision validation requires unique names.
            seen_ids = {}
            for rec in fetched_records:
                original = rec.id
                seen_ids[original] = seen_ids.get(original, 0) + 1
                if seen_ids[original] > 1:
                    rec.id = f'{original}_{seen_ids[original]}'
                    rec.name = rec.id
                if rec.description == original:
                    rec.description = rec.id
            output = io.StringIO()
            SeqIO.write(fetched_records, output, 'fasta')
            st.session_state.fetched_fasta_text = output.getvalue()
            st.session_state.fetch_results = statuses
            st.session_state.fetched_sequence_type = fetched_type
            st.success(f'Fetched {len(fetched_records)} record(s). Review the FASTA below before importing it into the analysis workflow.')
        else:
            st.session_state.fetched_fasta_text = ''
            st.session_state.fetch_results = statuses
            st.error('No records were fetched. Check the accession codes, selected NCBI database, and network connection.')
    if st.session_state.fetch_results:
        st.markdown('<div class="pv-section">Fetch summary</div>', unsafe_allow_html=True)
        st.dataframe(pd.DataFrame(st.session_state.fetch_results), use_container_width=True, hide_index=True)
    if st.session_state.fetched_fasta_text:
        fasta_out = st.session_state.fetched_fasta_text
        try:
            fetched_preview = list(SeqIO.parse(io.StringIO(fasta_out), 'fasta'))
            a, b, c = st.columns(3)
            a.metric('Records fetched', len(fetched_preview))
            b.metric('Total residues', f'{sum(len(r.seq) for r in fetched_preview):,}')
            c.metric('Sequence type', st.session_state.get('fetched_sequence_type', fetched_type))
        except Exception:
            pass
        st.text_area('FASTA preview', value=fasta_out, height=260)
        d1, d2 = st.columns(2)
        with d1:
            st.download_button('Download .fasta', data=fasta_out, file_name='phylovision_ncbi_sequences.fasta', mime='text/plain', use_container_width=True)
        with d2:
            if st.button('Import into Sequence Input', use_container_width=True):
                st.session_state.fasta_text = fasta_out
                st.session_state.sequence_type = st.session_state.get('fetched_sequence_type', fetched_type)
                st.session_state.tree = None
                st.session_state.tree_signature = None
                st.session_state.aligned_records = None
                st.session_state.dist_df = None
                navigate_to('Sequence Input')
                st.rerun()
        st.info('Important: NCBI returns the stored sequences; they are not automatically aligned. PhyloVision’s current similarity and tree tools require equal-length aligned sequences. Align the records first, then use Sequence Input and the analysis tools.')
    st.markdown('<div class="pv-card"><div class="pv-section">Troubleshooting</div><ul><li>Choose <b>Nucleotide</b> for DNA/RNA accessions or <b>Protein</b> for amino-acid accessions.</li><li>A valid accession in the wrong database may return no record.</li><li>Fetching requires Streamlit Cloud to have internet access; transient NCBI limits or network errors can be retried later.</li><li>Use records from the same gene/region or homologous protein family for meaningful evolutionary comparisons.</li></ul></div>', unsafe_allow_html=True)

elif page == 'Sequence Input':
    st.markdown('<div class="pv-title">Sequence Input</div><div class="pv-muted">Load a real plant rbcL or mammal COI example, upload a FASTA file, or paste sequences. Validation updates automatically as you edit.</div>',unsafe_allow_html=True)
    demo_a,demo_b=st.columns(2)
    with demo_a:
        if st.button('Load plant rbcL example', type='primary', key='input_load_plant_example', use_container_width=True):
            try:
                with st.spinner('Fetching plant rbcL records from NCBI…'):
                    fetch_plant_teaching_example(st.session_state.get('ncbi_email',''))
                st.rerun()
            except Exception as exc:
                st.error(f'Could not load the plant dataset: {exc}. Check your connection or use Accession → FASTA.')
    with demo_b:
        if st.button('Load mammal COI example', key='input_load_animal_example', use_container_width=True):
            try:
                with st.spinner('Fetching mammalian COI regions from NCBI…'):
                    fetch_animal_coi_example(st.session_state.get('ncbi_email',''))
                st.rerun()
            except Exception as exc:
                st.error(f'Could not load the mammal dataset: {exc}. Check your connection or use Accession → FASTA.')
    left,right=st.columns([1.15,1])
    with left:
        typ=st.selectbox('Sequence type',['DNA','Protein'],index=0 if st.session_state.sequence_type=='DNA' else 1)
        st.session_state.sequence_type=typ
        uploaded=st.file_uploader('Upload FASTA',type=['fasta','fa','fna','faa','txt'])
        if uploaded is not None:
            st.session_state.fasta_text=uploaded.getvalue().decode('utf-8',errors='replace')
        st.session_state.fasta_text=st.text_area('Or paste FASTA text',value=st.session_state.fasta_text,height=260,help='FASTA format: each record begins with > followed by a unique sequence name.')

    with right:
        if st.session_state.get('demo_source') and st.session_state.get('fasta_text') == st.session_state.get('demo_fasta_snapshot'):
            st.info('Teaching dataset source: ' + st.session_state.demo_source + '. Marker records are aligned to a common reference region for this demonstration; use a curated alignment for research.')
        st.markdown('<div class="pv-card"><div class="pv-section">FASTA checklist</div><ul><li>Use a header starting with <code>&gt;</code>.</li><li>Use unique, readable sequence names.</li><li>Use homologous sequences from a documented source for real analysis.</li><li>Sequences must be aligned and equal in length for this version.</li></ul></div>',unsafe_allow_html=True)
        try:
            recs=current_records(); st.success(f'Validation passed: {len(recs)} sequences are ready for the current alignment-based tools.'); st.metric('Sequences detected',len(recs)); st.metric('Sequence length',f'{len(recs[0].seq)} positions'); gaps=sum(str(r.seq).count('-') for r in recs); ambiguous=sum(sum(ch in 'N?X' for ch in str(r.seq).upper()) for r in recs); st.caption(f'Quality preview: {gaps} gap characters · {ambiguous} N/?/X ambiguous characters. Validation preview only; equal lengths do not prove a correct alignment.')
        except Exception as e: st.error(str(e))

elif page == 'Alignment Explorer':
    st.markdown('<div class="pv-title">Alignment Explorer</div><div class="pv-muted">Compare the sequences column by column. Colored letters make conserved and variable positions easier to spot.</div>',unsafe_allow_html=True)
    try:
        records=current_records(); lengths={len(r.seq) for r in records}
        if len(lengths)!=1: st.error('Sequences are different lengths. Please provide an alignment before using this view.')
        else:
            block_options = list(range(1, max(2, len(records[0].seq)+1), 60))
            selected_start = st.selectbox('Alignment block (60 positions)', block_options, format_func=lambda v: f'{v}–{min(v+59,len(records[0].seq))}', key='alignment_block')
            html,shown=alignment_html(records, start=selected_start-1, max_cols=60)
            st.markdown('<div class="pv-card"><div class="pv-section">Multiple sequence alignment</div><div class="pv-muted">Showing positions '+str(selected_start)+'–'+str(selected_start+shown-1)+'. A/T/G/C use a colour-blind-friendly palette. Conservation: * fully conserved, : at least 75% agreement, . variable. Consensus ties are shown as N.</div><br>'+html+'</div>',unsafe_allow_html=True)
            seqs=np.array([list(str(r.seq)) for r in records]); ncols=seqs.shape[1]
            conserved=[]; variable=[]
            for j in range(ncols):
                chars=[x for x in seqs[:,j] if x not in '-N?']; unique=set(chars)
                if len(unique)<=1: conserved.append(j+1)
                else: variable.append(j+1)
            a,b,c=st.columns(3); a.metric('Sequences',len(records)); b.metric('Alignment length',f'{ncols} bp' if st.session_state.sequence_type=='DNA' else f'{ncols} aa'); c.metric('Variable positions',len(variable))
            st.markdown('<div class="pv-card"><div class="pv-section">How to read this alignment</div><p>Each vertical column represents a position being compared across sequences. A column with the same character in most sequences is relatively conserved; a column with multiple characters is variable. Gaps can represent insertions or deletions introduced during alignment.</p><p class="pv-muted">Important: equal sequence lengths do not prove that the sequences are correctly aligned. Use a trusted alignment tool and check the biological context for research-quality work.</p></div>',unsafe_allow_html=True)
            st.download_button('Download aligned FASTA',data=''.join(f'>{r.id}\n{str(r.seq)}\n' for r in records),file_name='phylovision_alignment.fasta',mime='text/plain')
    except Exception as e: st.warning(str(e))

elif page == 'Phylogenetic Tree':
    st.markdown('<div class="pv-title">Phylogenetic Tree</div><div class="pv-muted">Build and explore a distance-based tree from an existing alignment.</div>',unsafe_allow_html=True)
    try:
        records=current_records()
        if len(records) < 3:
            st.error('Tree construction requires at least 3 sequences. Sequence Similarity can compare two or more.')
        elif len({len(r.seq) for r in records}) != 1:
            st.error('Sequences must have equal length and be aligned before tree construction.')
        else:
            x,y=st.columns([1,2.3])
            with x:
                method = st.selectbox('Tree-building method', ['Neighbor Joining','UPGMA'], help='Neighbor Joining is unrooted; UPGMA assumes a constant rate of evolution.')
                reps = st.slider('Bootstrap replicates', 0, 200, 100, step=25, help='100 is the default. Support is estimated by resampling aligned columns and rebuilding trees; 0 disables it.')
                st.markdown('<div class="pv-card"><div class="pv-section">Distance calculation</div><div class="pv-muted">Uncorrected p-distance based on observed identity. Gap/ambiguous handling is simplified for teaching use.</div></div>',unsafe_allow_html=True)
                if st.button('Build phylogenetic tree',use_container_width=True):
                    try:
                        with st.spinner('Building tree and evaluating resampled alignments…' if reps else 'Building tree…'):
                            if reps:
                                tree, support = bootstrap_tree(records, method, reps, sequence_type=st.session_state.sequence_type)
                            else:
                                tree, support = build_tree(records, method, st.session_state.sequence_type), {}
                        st.session_state.tree=tree
                        st.session_state.aligned_records=records
                        st.session_state.dist_df=make_distance_matrix(records, st.session_state.sequence_type)
                        st.session_state.bootstrap_support=support
                        st.session_state.tree_method=method
                        st.session_state.tree_signature=(st.session_state.fasta_text,st.session_state.sequence_type,method,reps)
                        st.success('Tree built successfully.')
                    except Exception as e:
                        st.error(f'Could not build tree: {e}')
                if st.session_state.tree is not None and st.session_state.tree_signature and st.session_state.tree_signature[0] == st.session_state.fasta_text and st.session_state.tree_signature[1] == st.session_state.sequence_type:
                    newick_io=io.StringIO(); Phylo.write(st.session_state.tree,newick_io,'newick')
                    st.download_button('Download tree (Newick)',newick_io.getvalue(),file_name='phylovision_tree.nwk',mime='text/plain',use_container_width=True)
                    try:
                        report = pdf_report(records, st.session_state.tree, st.session_state.tree_method, st.session_state.bootstrap_support, st.session_state.sequence_type)
                        st.download_button('Download analysis report (PDF)',report,file_name='phylovision_report.pdf',mime='application/pdf',use_container_width=True)
                    except Exception as e:
                        st.caption(f'PDF report unavailable: {e}')
            with y:
                if st.session_state.tree is not None and st.session_state.tree_signature and st.session_state.tree_signature[0] == st.session_state.fasta_text and st.session_state.tree_signature[1] == st.session_state.sequence_type:
                    tree_layout = st.segmented_control('Tree layout', options=['Radial garden','Rectangular'], default='Radial garden', key='tree_layout')
                    tree_fig = radial_tree_figure(st.session_state.tree) if tree_layout == 'Radial garden' else rectangular_tree_figure(st.session_state.tree)
                    st.plotly_chart(tree_fig,use_container_width=True,config={'displaylogo':False,'scrollZoom':True})
                    st.caption('Scroll to zoom • Drag to pan • Double-click to reset. Branch lengths represent estimated sequence change, not time.')
                    support = st.session_state.bootstrap_support
                    if support:
                        st.markdown('**Bootstrap split support**')
                        support_df = pd.DataFrame([{'Taxa on smaller side of split': ', '.join(k), 'Support (%)': round(v,1)} for k,v in sorted(support.items(),key=lambda item:item[1],reverse=True)])
                        st.dataframe(support_df,use_container_width=True,hide_index=True)
                        st.caption('Support is the percentage of column-resampled trees recovering that split; it is not the probability that a clade is true.')
                else:
                    if st.session_state.tree is not None:
                        st.info('Your input has changed since this tree was built. Build the tree again to refresh the result.')
                    st.markdown('<div class="pv-card" style="min-height:310px;display:flex;flex-direction:column;justify-content:center;text-align:center"><div style="font-size:3rem">🌿</div><div class="pv-section">Your tree will appear here</div><div class="pv-muted">Review the input, choose a method, then build your tree.</div></div>',unsafe_allow_html=True)
    except Exception as e: st.warning(str(e))

elif page == 'Evolution Lens':
    st.markdown('<div class="pv-title">Evolution Lens</div><div class="pv-muted">Connect tree relationships to the sequence positions being compared.</div>',unsafe_allow_html=True)
    try:
        records=current_records()
        if len({len(r.seq) for r in records}) != 1:
            st.error('Provide equal-length aligned sequences to compare positions.')
        else:
            names=[r.id for r in records]
            c1,c2=st.columns(2)
            with c1: first=st.selectbox('First sequence',names,index=0,key='lens_first')
            with c2: second=st.selectbox('Second sequence',names,index=1 if len(names)>1 else 0,key='lens_second')
            if first == second:
                st.info('Choose two different sequences to inspect their differences.')
            else:
                ra=next(r for r in records if r.id==first); rb=next(r for r in records if r.id==second)
                diffs=[]; compared=0; matches=0
                excluded = set('-?.NRYKMSWBDHVX') if st.session_state.sequence_type == 'DNA' else set('-?.XBZJ*')
                for pos,(a,b) in enumerate(zip(str(ra.seq).upper(),str(rb.seq).upper()),start=1):
                    if a in excluded or b in excluded: continue
                    compared += 1
                    if a == b: matches += 1
                    else: diffs.append({'Alignment position':pos,first:a,second:b,'Change':f'{a} → {b}'})
                identity=(matches/compared*100) if compared else float('nan')
                a,b,c=st.columns(3); a.metric('Compared positions',compared); b.metric('Differences',len(diffs)); c.metric('Pairwise identity',f'{identity:.2f}%' if compared else 'N/A')
                if diffs:
                    st.markdown('**Observed differences**')
                    st.dataframe(pd.DataFrame(diffs),use_container_width=True,hide_index=True)
                    st.download_button('Export differences (CSV)',pd.DataFrame(diffs).to_csv(index=False),file_name='phylovision_sequence_differences.csv',mime='text/csv')
                else:
                    st.success('No differences were found among the comparable positions.')
                st.caption('This view reports observed differences in your supplied alignment. It does not prove that any one mutation caused an evolutionary event.')
    except Exception as e:
        st.warning(str(e))

elif page == 'Understand My Tree':
    st.markdown('<div class="pv-title">Understand My Tree</div><div class="pv-muted">A plain-language guide to the tree you actually generated.</div>',unsafe_allow_html=True)
    if st.session_state.tree is None or not st.session_state.tree_signature or st.session_state.tree_signature[0] != st.session_state.fasta_text or st.session_state.tree_signature[1] != st.session_state.sequence_type:
        st.info('Build or refresh the tree in the Phylogenetic Tree tab using your current input before interpreting it.')
    else:
        tree=st.session_state.tree; terms=tree.get_terminals(); names=[t.name for t in terms]
        selected=st.selectbox('Choose a sequence to inspect',names)
        chosen=next(t for t in terms if t.name==selected)
        others=sorted([(tree.distance(chosen,t),t.name) for t in terms if t.name!=selected])
        a,b=st.columns([1,1])
        with a:
            st.markdown('<div class="pv-card"><div class="pv-section">Selected sequence</div><div style="font-size:1.25rem;font-weight:700;color:#d8b0ff">'+selected+'</div><div class="pv-muted">Closest sequence by tree path</div><div style="font-size:1.1rem;margin-top:6px">'+others[0][1]+'</div><div class="pv-muted">Patristic distance: '+f'{others[0][0]:.4f}'+'</div></div>',unsafe_allow_html=True)
        with b:
            st.markdown('<div class="pv-card"><div class="pv-section">Interpretation guide</div><ul><li><b>Tips</b> are the sampled sequences.</li><li><b>Nodes</b> represent inferred branching points.</li><li><b>Branch lengths</b> represent estimated sequence change in this distance-based analysis.</li><li><b>Support values</b> are not calculated by this version.</li></ul></div>',unsafe_allow_html=True)
        st.markdown('<div class="pv-card"><div class="pv-section">Key findings</div><ul style="padding-left:1.25rem;margin-bottom:0">'+''.join('<li style="margin:.45rem 0;color:#393148">'+finding.replace('**','')+'</li>' for finding in explain_tree(tree,names))+'</ul></div>',unsafe_allow_html=True)
        st.info('Interpretation is limited to the input sequences and method. Bootstrap support, when calculated, describes split recovery under column resampling; it is not the probability that a clade is true.')

elif page == 'Sequence Similarity':
    st.markdown('<div class="pv-kicker">COMPARE · QUANTIFY · EXPORT</div><div class="pv-title">Sequence similarity</div><div class="pv-muted">Pairwise identity across comparable alignment positions. This view is independent of whether a tree has already been built.</div>',unsafe_allow_html=True)
    try:
        records = current_records()
        sim = make_similarity_matrix(records, st.session_state.sequence_type)
        finite = sim.to_numpy()[np.isfinite(sim.to_numpy())]
        if len(finite) == 0:
            st.warning('No comparable positions were found. Check that the sequences contain valid, non-ambiguous residues.')
        else:
            upper = sim.where(np.triu(np.ones(sim.shape), k=1).astype(bool))
            pair_values = upper.stack().dropna()
            if len(pair_values):
                vals = pair_values.to_numpy(dtype=float)
                names = list(pair_values.index.names)
                pair_items = [(float(v), str(i[0]), str(i[1])) for i, v in pair_values.items()]
                most_value = max(x[0] for x in pair_items); least_value = min(x[0] for x in pair_items)
                most_ties = [(a,b) for v,a,b in pair_items if math.isclose(v,most_value,rel_tol=1e-9,abs_tol=1e-9)]
                least_ties = [(a,b) for v,a,b in pair_items if math.isclose(v,least_value,rel_tol=1e-9,abs_tol=1e-9)]
                most_text = '; '.join(f'{a} ↔ {b}' for a,b in most_ties[:3]) + (f' (+{len(most_ties)-3} tied pairs)' if len(most_ties)>3 else '')
                least_text = '; '.join(f'{a} ↔ {b}' for a,b in least_ties[:3]) + (f' (+{len(least_ties)-3} tied pairs)' if len(least_ties)>3 else '')
                m1,m2,m3 = st.columns(3)
                m1.metric('Mean identity', f'{float(np.mean(vals)):.2f}%')
                m2.metric('Minimum identity', f'{float(np.min(vals)):.2f}%')
                m3.metric('Maximum identity', f'{float(np.max(vals)):.2f}%')
                st.markdown(f'<div class="pv-card"><b>Most similar pair(s):</b> {most_text} ({most_value:.2f}%)<br><b>Least similar pair(s):</b> {least_text} ({least_value:.2f}%)<br><span class="pv-muted">{len(pair_values)} unique pairs · ties are reported; identity excludes gaps and alphabet-appropriate ambiguity/missing symbols.</span></div>', unsafe_allow_html=True)
            fig = go.Figure(data=go.Heatmap(z=sim.values, x=sim.columns, y=sim.index,
                colorscale=[[0,'#f2edf8'],[0.5,'#d7c8e9'],[1,'#a78acb']], zmin=0,zmax=100,connectgaps=False,
                colorbar=dict(title='Identity %',tickfont=dict(color='#514a60'),outlinewidth=0),
                hovertemplate='%{y} vs %{x}<br>Identity: %{z:.2f}%<extra></extra>',
                text=np.where(np.isfinite(sim.values), np.vectorize(lambda v: f'{v:.1f}%' if np.isfinite(v) else 'No data')(sim.values), 'No data'),
                texttemplate='%{text}', textfont=dict(color='#302a40',size=13)))
            finite_pair_values = pair_values.to_numpy(dtype=float) if len(pair_values) else finite
            zmin = float(np.nanmin(finite))
            zmax = float(np.nanmax(finite))
            if zmax <= zmin: zmin, zmax = max(0.0, zmin - 1.0), min(100.0, zmax + 1.0)
            fig.update_layout(template='plotly_white',height=max(360,len(records)*54),margin=dict(l=90,r=30,t=25,b=100),paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)',font=dict(color='#302a40',size=12),xaxis=dict(tickangle=-35,side='bottom',tickfont=dict(color='#302a40',size=11)),yaxis=dict(autorange='reversed',tickfont=dict(color='#302a40',size=11)))
            fig.data[0].zmin = zmin
            fig.data[0].zmax = zmax
            st.plotly_chart(fig,use_container_width=True,config={'displaylogo':False})
            with st.expander('Inspect pairwise identity values', expanded=True):
                st.dataframe(sim.style.format('{:.2f}%',na_rep='No comparable sites'),use_container_width=True)
            out=sim.to_csv(index=True,na_rep='')
            st.download_button('Export similarity matrix · CSV',out,file_name='phylovision_similarity.csv',mime='text/csv')
            st.markdown('<div class="pv-card"><div class="pv-section">How to read this</div><div class="pv-muted">100% means all comparable positions match; lower values indicate more observed differences. “No comparable sites” means every position in that pair was excluded. Identity is descriptive, not a probability of common ancestry or a bootstrap score.</div></div>',unsafe_allow_html=True)
    except Exception as e:
        st.error(f'Could not calculate sequence similarity: {e}')

else: # Creators & About
    st.markdown('<div class="pv-title">Creators & About</div><div class="pv-muted">The people and purpose behind PhyloVision.</div>',unsafe_allow_html=True)
    st.markdown('<div class="pv-card"><div class="pv-kicker">Project creator</div><div style="font-family:Manrope;font-size:1.5rem;font-weight:800;color:#79639d;margin-top:4px">Anureet Dutta</div><div class="pv-muted">Creator & project lead</div><hr><div class="pv-kicker">Co-creators</div><div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:12px;margin-top:12px">'+''.join(f'<div style="padding:14px 15px;border-radius:12px;background:#f5f1fa;border:1px solid #e2d9ef"><div style="font-weight:700;color:#443653">{name}</div><div class="pv-muted">Co-creator</div></div>' for name in CO_CREATORS)+'</div></div>',unsafe_allow_html=True)
    st.markdown('<div class="pv-card"><div class="pv-section">About the project</div><p>PhyloVision is an educational bioinformatics dashboard designed to connect multiple sequence alignment with phylogenetic tree construction and interpretation. It brings sequence input, an alignment view, a distance-based tree and pairwise identity summaries into one interface.</p><p><b>Learning outcomes:</b> FASTA handling, sequence validation, alignment inspection, pairwise similarity, distance-based tree construction and cautious biological interpretation.</p><p class="pv-muted">User-supplied sequences must already be aligned and equal length. Teaching examples fetch real plant rbcL or mammalian COI records and use a simple reference-guided teaching alignment for a 450-position region. Trees use identity-based uncorrected p-distance; bootstrap split support is available when resampling is run. Protein input is accepted for comparison, but this simple distance approach is not a substitute for a validated protein-specific evolutionary model. For research, use curated alignments and document sequence accessions and settings.</p></div>',unsafe_allow_html=True)
    st.markdown('<div class="pv-card"><div class="pv-section">Credits & acknowledgement</div><p>Built as a student biotechnology software project, with a focus on making sequence-based evolutionary analysis easier to explore and explain.</p><div class="pv-muted">Better data. Clearer relationships. A deeper understanding of life.</div></div>',unsafe_allow_html=True)

st.markdown('<hr><div style="text-align:center;color:#5f596d;font-size:.9rem;padding:8px 0">PhyloVision · From Sequences to Evolution · Educational use</div>',unsafe_allow_html=True)
