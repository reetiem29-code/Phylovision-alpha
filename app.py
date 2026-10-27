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
[data-testid="stProgressBar"] > div > div { background-color:#9a83bf; }
</style>
''', unsafe_allow_html=True)

# Demo sequences are clearly labelled teaching examples; users can replace them.
DEMO_FASTA = '''>Human_demo\nATGCTAGCTAGCTACGATCGATCGATGCTAGCTAGCTAACGTTAGCTAGCTAGGCTAACGATCGATCGATGCTAGC\n>Chimp_demo\nATGCTAGCTAGCTACGATCGATCGATGCTAGCTAGCTAACGTTAGCTAGCTAGGCTAACGATCGATCGATGCTAGT\n>Gorilla_demo\nATGCTAGCTAGCTACGATCGATCGATGCTAGCTAGCTAACGTTAGCTAGCTAGGCTAACGATCGATCAATGCTAGT\n>Mouse_demo\nATGATAGCTTGCTACGATCAATCGATGCTAGATAGCTAACGTTAGCTAGTTAGGCTAACGATCAATCGATGCTAGT\n>Rat_demo\nATGATAGCTTGCTACGATCAATCGATGCTAGATAGCTAACGTTAGCTAGTTAGGCTAACGATCAATCGATGCTAGC'''
BASE_COLORS = {'A':'#65bd8b','T':'#ed7e88','G':'#f1bd65','C':'#67b8ee','U':'#b89ce9','-':'#59627f','N':'#8b8fa7'}
CREATOR = 'Anureet Dutta'
CO_CREATORS = ['Sreejeeta Kundu','Aishwarya Banik','Soham Pariyal','Ram Argha Nayak']

# Header with the supplied project logo.
h1, h2 = st.columns([1.5, 3.6], vertical_alignment='center')
with h1:
    if LOGO.exists(): st.image(str(LOGO), width=280)
    else: st.markdown('<div class="pv-title">🧬 PhyloVision</div>', unsafe_allow_html=True)
with h2:
    st.markdown('<div class="pv-kicker">From Sequences to Evolution</div><div style="font-size:1.55rem;font-weight:700;color:#393148;font-family:Manrope">Biological relationships, with room to breathe.</div><div class="pv-muted">A calm, precise workspace for sequence comparison and evolutionary exploration.</div>', unsafe_allow_html=True)
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
if 'fasta_text' not in st.session_state: st.session_state.fasta_text = DEMO_FASTA
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
    st.session_state.page = 'Home'

with st.sidebar:
    st.markdown('<div style="font-family:Manrope,sans-serif;font-size:1.28rem;font-weight:800;letter-spacing:-.035em">✳ PhyloVision</div><div style="color:#8772ad;font-size:.68rem;letter-spacing:.15em;font-weight:700;margin:.15rem 0 .9rem">BOTANICAL BIOINFORMATICS</div>', unsafe_allow_html=True)
    st.markdown('<div class="pv-side-label">QUICK ACTION SEARCH</div>', unsafe_allow_html=True)
    search_query = st.text_input('Search tools', placeholder='Find a tool or action…', label_visibility='collapsed', key='quick_action_search')
    if search_query.strip():
        matches = [name for name in ALL_PAGES if search_query.lower() in name.lower() or search_query.lower() in PAGE_HINTS[name].lower()]
        if matches:
            chosen_match = st.selectbox('Matching actions', matches, label_visibility='collapsed', key='quick_action_match')
            if st.button('Open selected tool →', use_container_width=True, key='open_quick_action'):
                st.session_state.page = chosen_match
                st.rerun()
        else:
            st.caption('No matching tool. Try “tree”, “FASTA”, “similarity”, or “alignment”.')
    st.markdown('<div class="pv-side-divider"></div><div class="pv-side-label">GUIDED LAB</div><div class="pv-muted" style="font-size:.82rem;margin:.25rem 0 .65rem">Follow the steps in order, or jump ahead whenever you are ready.</div>', unsafe_allow_html=True)
    guided_steps = [
        ('1 · Prepare sequences', 'Sequence Input', 'Paste, upload, or load demo data'),
        ('2 · Fetch by accession', 'Accession → FASTA', 'Optional: retrieve records from NCBI'),
        ('3 · Inspect alignment', 'Alignment Explorer', 'Check lengths, gaps, and positions'),
        ('4 · Build a tree', 'Phylogenetic Tree', 'Infer relationships from aligned data'),
        ('5 · Explore findings', 'Sequence Similarity', 'Compare and interpret results'),
    ]
    for label, target, hint in guided_steps:
        active = st.session_state.page == target
        if st.button(('●  ' if active else '○  ') + label, key='guided_'+target, use_container_width=True, help=hint, type='primary' if active else 'secondary'):
            st.session_state.page = target
            st.rerun()
        st.markdown(f'<div class="pv-step-hint">{hint}</div>', unsafe_allow_html=True)
    st.markdown('<div class="pv-side-divider"></div><div class="pv-side-label">TOOL DOCK · DIRECT ACCESS</div>', unsafe_allow_html=True)
    dock_pages = ['Home', 'Sequence Input', 'Accession → FASTA', 'Alignment Explorer', 'Phylogenetic Tree', 'Sequence Similarity', 'Evolution Lens', 'Understand My Tree', 'Creators & About']
    dock_choice = st.selectbox('Open a tool directly', dock_pages, index=dock_pages.index(st.session_state.page) if st.session_state.page in dock_pages else 0, label_visibility='collapsed', key='tool_dock_select')
    if st.button('Go to tool ↗', use_container_width=True, key='tool_dock_go'):
        st.session_state.page = dock_choice
        st.rerun()

    # Evidence-based readiness: only count milestones that can be verified from current state.
    has_input = bool(st.session_state.get('fasta_text', '').strip())
    has_alignment = st.session_state.get('aligned_records') is not None
    has_tree = st.session_state.get('tree') is not None
    done = sum([has_input, has_alignment, has_tree])
    st.markdown('<div class="pv-side-divider"></div><div class="pv-side-label">LAB READINESS</div>', unsafe_allow_html=True)
    st.progress(done / 3, text=f'{done}/3 core milestones')
    st.markdown(f'<div class="pv-readiness">{"✓" if has_input else "○"} Sequence input<br>{"✓" if has_alignment else "○"} Alignment available<br>{"✓" if has_tree else "○"} Tree built</div><div class="pv-step-hint">Readiness reflects this session’s actual state; it does not certify biological validity.</div>', unsafe_allow_html=True)
    st.markdown('<div class="pv-side-divider"></div><div class="pv-muted" style="font-size:.75rem">Educational analysis · Interpret results in light of sequence quality and method.</div>', unsafe_allow_html=True)

page = st.session_state.page

if 'fasta_text' not in st.session_state: st.session_state.fasta_text = DEMO_FASTA
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


def comparable_pair(a, b):
    # Exclude gaps and ambiguous/missing symbols from BOTH sides of a pair.
    excluded = set('-?.NX')
    pairs = [(x, y) for x, y in zip(str(a).upper(), str(b).upper())
             if x not in excluded and y not in excluded]
    compared = len(pairs)
    matches = sum(x == y for x, y in pairs)
    return matches, compared


def pairwise_p_distance(a,b):
    matches, compared = comparable_pair(a, b)
    if compared == 0: return np.nan
    return (compared - matches) / compared


def make_similarity_matrix(records):
    """Pairwise percent identity; NaN means no comparable positions."""
    names = [r.id for r in records]
    mat = np.full((len(records), len(records)), np.nan, dtype=float)
    for i, record in enumerate(records):
        _, compared = comparable_pair(record.seq, record.seq)
        if compared: mat[i, i] = 100.0
        for j in range(i + 1, len(records)):
            matches, compared = comparable_pair(record.seq, records[j].seq)
            value = (100.0 * matches / compared) if compared else np.nan
            mat[i, j] = mat[j, i] = value
    return pd.DataFrame(mat, index=names, columns=names)


def make_distance_matrix(records):
    names=[r.id for r in records]
    mat=np.zeros((len(records),len(records)))
    for i in range(len(records)):
        for j in range(i+1,len(records)):
            d=pairwise_p_distance(records[i].seq,records[j].seq)
            mat[i,j]=mat[j,i]=d if np.isfinite(d) else 1.0
    return pd.DataFrame(mat,index=names,columns=names)


def build_tree(records, model='Neighbor Joining'):
    """Build a tree from an explicit Biopython MultipleSeqAlignment.

    The original bug passed a plain list of SeqRecord objects to
    DistanceCalculator.get_distance(); Biopython requires an Alignment object.
    """
    alignment = MultipleSeqAlignment([r[:] for r in records])
    calculator = DistanceCalculator('identity', skip_letters='N?-X')
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


def bootstrap_tree(records, model='Neighbor Joining', replicates=100, seed=17):
    """Column-resampling bootstrap; returns the original tree and split support %.

    This is a standard non-parametric site bootstrap for an already-aligned
    dataset. It is intentionally capped by the UI for a student-hosted app.
    """
    base_tree = build_tree(records, model)
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
        replicate_tree = build_tree(sampled, model)
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
        fig.add_trace(go.Scatter(x=[xmap[parent],xmap[parent]],y=[min(ys),max(ys)],mode='lines',line=dict(color='#7e8ab9',width=2),hoverinfo='skip',showlegend=False))
        for c in children:
            color=palette[terminals.index(c)%len(palette)] if c.is_terminal() else '#8b83c8'
            fig.add_trace(go.Scatter(x=[xmap[parent],xmap[c]],y=[ymap[c],ymap[c]],mode='lines',line=dict(color=color,width=2.4),hoverinfo='skip',showlegend=False))
    maxx=max(xmap.values()) if xmap else 1
    for t in terminals:
        fig.add_trace(go.Scatter(x=[xmap[t]],y=[ymap[t]],mode='markers+text',text=[t.name],textposition='middle right',textfont=dict(color='#393448',size=11),marker=dict(size=7,color=palette[terminals.index(t)%len(palette)]),hovertemplate=f'<b>{t.name}</b><br>Root distance: {xmap[t]:.4f}<extra></extra>',showlegend=False))
    fig.update_layout(height=max(430,n*42),margin=dict(l=10,r=130,t=10,b=10),paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)',xaxis=dict(title='Accumulated branch length (relative evolutionary change)',showgrid=True,gridcolor='rgba(130,145,190,.15)',zeroline=False,color='#6e687c',rangemode='tozero'),yaxis=dict(visible=False,autorange='reversed'),font=dict(family='DM Sans',color='#393448'),showlegend=False)
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
            fig.add_trace(go.Scatter(x=[px,cx], y=[py,cy], mode='lines', line=dict(color=color,width=2.2), hoverinfo='skip', showlegend=False))
    for term in terminals:
        x, y = xy(term); a = angle_map[term]
        label_radius = radius_map[term] + max_r * .07
        lx, ly = label_radius * math.cos(a), label_radius * math.sin(a)
        fig.add_trace(go.Scatter(x=[x],y=[y],mode='markers',marker=dict(size=7,color=palette[terminals.index(term)%len(palette)],line=dict(color='#e6eddc',width=.8)),name=term.name,hovertemplate=f'<b>{term.name}</b><br>Root distance: {radius_map[term]:.4f}<extra></extra>',showlegend=False))
        fig.add_annotation(x=lx,y=ly,text=term.name,showarrow=False,font=dict(size=10,color='#393448'),xanchor='left' if math.cos(a)>=0 else 'right',yanchor='middle')
    limit = max_r * 1.28
    fig.update_layout(height=max(520,n*34),margin=dict(l=25,r=25,t=25,b=25),paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)',font=dict(family='DM Sans',color='#393448'),showlegend=False,xaxis=dict(visible=False,range=[-limit,limit],scaleanchor='y',scaleratio=1),yaxis=dict(visible=False,range=[-limit,limit]),hovermode='closest')
    return fig


def alignment_html(records, max_cols=90):
    max_len=min(max(len(r.seq) for r in records),max_cols)
    html='<div style="overflow-x:auto"><table style="border-collapse:separate;border-spacing:2px;font:12px monospace;color:#f4f1ff;width:max-content"><thead><tr><th style="text-align:left;position:sticky;left:0;background:#171e3a;padding:6px">Sequence</th>'
    for i in range(max_len):
        if i % 10 == 0: html+=f'<th colspan="1" style="color:#b9bfdc;padding:2px">{i+1}</th>'
        else: html+='<th style="padding:2px"></th>'
    html+='</tr></thead><tbody>'
    for r in records:
        html+=f'<tr><td style="padding:4px 8px;position:sticky;left:0;background:#171e3a;white-space:nowrap">{r.id}</td>'
        for ch in str(r.seq)[:max_len]:
            c=BASE_COLORS.get(ch,'#8b8fa7')
            html+=f'<td style="background:{c};color:#10162b;border-radius:3px;text-align:center;min-width:17px;padding:3px">{ch}</td>'
        html+='</tr>'
    html+='</tbody></table></div>'
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
        findings.append(f'**Closest pair in this tree:** {a} and {b} have the smallest patristic distance in this inferred tree ({d:.4f}). This is a result of the supplied sequences and method, not proof that one modern organism descended from the other.')
    root=tree.root
    if root.clades:
        largest=max(root.clades,key=lambda c:len(c.get_terminals()))
        if len(largest.get_terminals())>=2:
            members=', '.join(t.name for t in largest.get_terminals())
            findings.append(f'**One large grouping:** {members} appear together on one side of the current unrooted tree. Tree orientation/rooting affects how the grouping is displayed.')
    findings.append('**Branch lengths:** longer paths represent greater estimated sequence change under the distance calculation. They do not automatically represent elapsed time.')
    findings.append('**Confidence:** this version does not calculate bootstrap support. Do not interpret branch lengths or visual spacing as confidence values.')
    return findings


def current_records():
    records=parse_fasta(st.session_state.fasta_text)
    validate_records(records, st.session_state.sequence_type)
    return records

# Home
if page == 'Home':
    st.markdown('<div class="pv-card"><div class="pv-kicker">Welcome to your sequence workspace</div><div class="pv-title">From raw sequences to clearer relationships.</div><div class="pv-muted">PhyloVision brings sequence preparation, alignment inspection, similarity comparison and phylogenetic interpretation into one calm, student-friendly workflow.</div></div>',unsafe_allow_html=True)
    st.markdown('<div class="pv-accession-feature"><div class="pv-kicker">A faster way to get started</div><div class="pv-section" style="margin:.3rem 0">Have accession codes? Turn them into FASTA.</div><div class="pv-muted">Retrieve nucleotide or protein records from NCBI without leaving PhyloVision, preview the sequences, then bring them into your workflow. Find it in the sidebar under <b>Guided Lab → Fetch by accession</b>, or use <b>Quick Action Search</b> and type “accession”.</div></div>', unsafe_allow_html=True)
    if st.button('Open Accession → FASTA', key='home_accession_cta'):
        st.session_state.page = 'Accession → FASTA'
        st.rerun()
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
                st.markdown(f'<div class="pv-guide-item"><div class="pv-section">{item[0]}</div><div class="pv-muted">{item[1]}</div><div style="font-size:.78rem;color:#776396;margin-top:8px"><b>Find it:</b> {item[2]}</div></div>', unsafe_allow_html=True)
    st.markdown('<div class="pv-card" style="margin-top:14px"><div class="pv-section">Suggested beginner workflow</div><ol><li>Prepare sequences or fetch them using accession codes.</li><li>Inspect the alignment and confirm sequences are suitable for comparison.</li><li>Build a phylogenetic tree.</li><li>Explore similarity and interpret the result cautiously.</li></ol><div class="pv-muted">The sidebar readiness indicator reflects session state, not scientific validity. Demo sequences are illustrative teaching data, not verified reference sequences.</div></div>',unsafe_allow_html=True)

elif page == 'Accession → FASTA':
    st.markdown('<div class="pv-kicker">Sequence acquisition</div><div class="pv-title">Accession → FASTA</div><div class="pv-muted">A key PhyloVision shortcut: retrieve nucleotide or protein records from NCBI by accession code, preview the returned FASTA, and bring it into your existing workflow.</div>', unsafe_allow_html=True)
    st.markdown('<div class="pv-card"><div class="pv-section">Fetch records from NCBI</div><div class="pv-muted">Enter one accession per line. Choose the database explicitly; an accession by itself does not always tell us whether it represents DNA or protein.</div></div>', unsafe_allow_html=True)
    db_label = st.radio('NCBI database', ['Nucleotide (DNA/RNA)', 'Protein'], horizontal=True, key='accession_database')
    db_name = 'nucleotide' if db_label.startswith('Nucleotide') else 'protein'
    fetched_type = 'DNA' if db_name == 'nucleotide' else 'Protein'
    accession_text = st.text_area('Accession codes', placeholder='NM_000546.6\nNC_045512.2', height=115, help='Paste accession codes, one per line. Commas and semicolons are also accepted.')
    contact_email = st.text_input('Contact email (recommended by NCBI)', placeholder='you@example.com', help='Used only to identify requests to NCBI; it is not stored by PhyloVision.')
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
                st.success('Imported into PhyloVision session. Open Sequence Input to review and validate the records.')
        st.info('Important: NCBI returns the stored sequences; they are not automatically aligned. PhyloVision’s current similarity and tree tools require equal-length aligned sequences. Align the records first, then use Sequence Input and the analysis tools.')
    st.markdown('<div class="pv-card"><div class="pv-section">Troubleshooting</div><ul><li>Choose <b>Nucleotide</b> for DNA/RNA accessions or <b>Protein</b> for amino-acid accessions.</li><li>A valid accession in the wrong database may return no record.</li><li>Fetching requires Streamlit Cloud to have internet access; transient NCBI limits or network errors can be retried later.</li><li>Use records from the same gene/region or homologous protein family for meaningful evolutionary comparisons.</li></ul></div>', unsafe_allow_html=True)

elif page == 'Sequence Input':
    st.markdown('<div class="pv-title">Sequence Input</div><div class="pv-muted">Upload a FASTA file or paste sequences. Give each sequence a unique header.</div>',unsafe_allow_html=True)
    left,right=st.columns([1.15,1])
    with left:
        typ=st.selectbox('Sequence type',['DNA','Protein'],index=0 if st.session_state.sequence_type=='DNA' else 1)
        st.session_state.sequence_type=typ
        uploaded=st.file_uploader('Upload FASTA',type=['fasta','fa','fna','faa','txt'])
        if uploaded is not None:
            st.session_state.fasta_text=uploaded.getvalue().decode('utf-8',errors='replace')
        st.session_state.fasta_text=st.text_area('Or paste FASTA text',value=st.session_state.fasta_text,height=260,help='FASTA format: each record begins with > followed by a unique sequence name.')
        if st.button('Validate sequences',use_container_width=True):
            try:
                recs=current_records(); st.success(f'Validated {len(recs)} sequences. All have equal length ({len(recs[0].seq)} positions), suitable for the current analysis.')
            except Exception as e: st.error(str(e))
        if st.button('Load teaching example',use_container_width=True):
            st.session_state.fasta_text=DEMO_FASTA; st.session_state.sequence_type='DNA'; st.session_state.tree=None; st.rerun()
    with right:
        st.markdown('<div class="pv-card"><div class="pv-section">FASTA checklist</div><ul><li>Use a header starting with <code>&gt;</code>.</li><li>Use unique, readable sequence names.</li><li>Use homologous sequences from a documented source for real analysis.</li><li>Sequences must be aligned and equal in length for this version.</li></ul></div>',unsafe_allow_html=True)
        try:
            recs=current_records(); st.metric('Sequences detected',len(recs)); st.metric('Sequence length',f'{len(recs[0].seq)} positions'); gaps=sum(str(r.seq).count('-') for r in recs); ambiguous=sum(sum(ch in 'N?X' for ch in str(r.seq).upper()) for r in recs); st.caption(f'Quality preview: {gaps} gap characters · {ambiguous} N/?/X ambiguous characters. Validation preview only; equal lengths do not prove a correct alignment.')
        except Exception as e: st.warning(str(e))

elif page == 'Alignment Explorer':
    st.markdown('<div class="pv-title">Alignment Explorer</div><div class="pv-muted">Compare the sequences column by column. Colored letters make conserved and variable positions easier to spot.</div>',unsafe_allow_html=True)
    try:
        records=current_records(); lengths={len(r.seq) for r in records}
        if len(lengths)!=1: st.error('Sequences are different lengths. Please provide an alignment before using this view.')
        else:
            html,shown=alignment_html(records)
            st.markdown('<div class="pv-card"><div class="pv-section">Multiple sequence alignment</div><div class="pv-muted">Showing the first '+str(shown)+' positions. A = adenine, T = thymine, G = guanine, C = cytosine; gaps are shown as dashes.</div><br>'+html+'</div>',unsafe_allow_html=True)
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
                reps = st.slider('Bootstrap replicates', 0, 200, 0, step=25, help='0 disables bootstrap. Higher values take longer. Uses alignment-column resampling.')
                st.markdown('<div class="pv-card"><div class="pv-section">Distance calculation</div><div class="pv-muted">Uncorrected p-distance based on observed identity. Gap/ambiguous handling is simplified for teaching use.</div></div>',unsafe_allow_html=True)
                if st.button('Build phylogenetic tree',use_container_width=True):
                    try:
                        with st.spinner('Building tree and evaluating resampled alignments…' if reps else 'Building tree…'):
                            if reps:
                                tree, support = bootstrap_tree(records, method, reps)
                            else:
                                tree, support = build_tree(records, method), {}
                        st.session_state.tree=tree
                        st.session_state.aligned_records=records
                        st.session_state.dist_df=make_distance_matrix(records)
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
                for pos,(a,b) in enumerate(zip(str(ra.seq).upper(),str(rb.seq).upper()),start=1):
                    if a in '-N?X' or b in '-N?X': continue
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
        st.markdown('<div class="pv-card"><div class="pv-section">Key findings</div></div>',unsafe_allow_html=True)
        for finding in explain_tree(tree,names): st.markdown(f'- {finding}')
        st.warning('Interpretation is limited to the input sequences and the method. The tree does not prove direct ancestry, and this version does not calculate bootstrap confidence.')

elif page == 'Sequence Similarity':
    st.markdown('<div class="pv-kicker">COMPARE · QUANTIFY · EXPORT</div><div class="pv-title">Sequence similarity</div><div class="pv-muted">Pairwise identity across comparable alignment positions. This view is independent of whether a tree has already been built.</div>',unsafe_allow_html=True)
    try:
        records = current_records()
        sim = make_similarity_matrix(records)
        finite = sim.to_numpy()[np.isfinite(sim.to_numpy())]
        if len(finite) == 0:
            st.warning('No comparable positions were found. Check that the sequences contain valid, non-ambiguous residues.')
        else:
            upper = sim.where(np.triu(np.ones(sim.shape), k=1).astype(bool))
            pair_values = upper.stack().dropna()
            if len(pair_values):
                st.metric('Sequence pairs compared', len(pair_values))
                st.caption(f'{len(records)} sequences · Identity excludes gaps and ambiguous/missing symbols (-, ?, ., N, X).')
            fig = go.Figure(data=go.Heatmap(z=sim.values, x=sim.columns, y=sim.index,
                colorscale=[[0,'#3b5148'],[.5,'#8cae91'],[1,'#d9e8bf']], zmin=0,zmax=100,connectgaps=False,
                colorbar=dict(title='Identity %',tickfont=dict(color='#514a60'),outlinewidth=0),
                hovertemplate='%{y} vs %{x}<br>Identity: %{z:.2f}%<extra></extra>',
                text=np.where(np.isfinite(sim.values), np.vectorize(lambda v: f'{v:.1f}%' if np.isfinite(v) else 'No data')(sim.values), 'No data'),
                texttemplate='%{text}', textfont=dict(color='#17211b',size=11)))
            fig.update_layout(height=max(360,len(records)*54),margin=dict(l=10,r=18,t=12,b=75),paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)',font=dict(color='#393448',size=11),xaxis=dict(tickangle=-35,side='bottom'),yaxis=dict(autorange='reversed'))
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
    st.markdown('<div class="pv-card"><div class="pv-section">About the project</div><p>PhyloVision is an educational bioinformatics dashboard designed to connect multiple sequence alignment with phylogenetic tree construction and interpretation. It brings sequence input, an alignment view, a distance-based tree and pairwise identity summaries into one interface.</p><p><b>Learning outcomes:</b> FASTA handling, sequence validation, alignment inspection, pairwise similarity, distance-based tree construction and cautious biological interpretation.</p><p class="pv-muted">Current limitations: sequences must already be aligned and equal length; the tree uses Neighbor Joining with uncorrected p-distance; no bootstrap support is calculated; protein input is accepted for comparison, but this simple distance approach is not a substitute for a validated protein-specific evolutionary model. For research, use suitable validated tools and document sequence accessions and settings.</p></div>',unsafe_allow_html=True)
    st.markdown('<div class="pv-card"><div class="pv-section">Credits & acknowledgement</div><p>Built as a student biotechnology software project, with a focus on making sequence-based evolutionary analysis easier to explore and explain.</p><div class="pv-muted">Better data. Clearer relationships. A deeper understanding of life.</div></div>',unsafe_allow_html=True)

st.markdown('<hr><div style="text-align:center;color:#9ea8cc;font-size:.8rem;padding:8px 0">PhyloVision · From Sequences to Evolution · Educational use</div>',unsafe_allow_html=True)
