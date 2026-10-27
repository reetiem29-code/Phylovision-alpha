from pathlib import Path
import io, re, math
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from Bio import SeqIO, AlignIO, Phylo
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio.Align import MultipleSeqAlignment
from Bio.Phylo.TreeConstruction import DistanceCalculator, DistanceTreeConstructor

APP_DIR = Path(__file__).parent
LOGO = APP_DIR / 'phyloVision_logo.png'

st.set_page_config(page_title='PhyloVision | From Sequences to Evolution', page_icon='🧬', layout='wide', initial_sidebar_state='expanded')

st.markdown('''
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap');
:root { --pv-bg:#0c1022; --pv-panel:#151b35; --pv-panel2:#1b2342; --pv-border:#343e65; --pv-purple:#bb83ec; --pv-pink:#e58dbd; --pv-text:#f2f1ff; --pv-muted:#b9bfdc; }
.stApp { background: radial-gradient(ellipse at 10% 0%, #27264e 0%, #11162c 36%, #0b1021 78%); color:var(--pv-text); font-family:'DM Sans',sans-serif; }
[data-testid="stHeader"] { background:transparent; }
[data-testid="stSidebar"] { background:linear-gradient(180deg,#171b37,#10152b); border-right:1px solid #30385c; }
[data-testid="stSidebar"] * { color:#eeeaff; }
.block-container { padding-top:1.1rem; padding-bottom:2rem; max-width:1600px; }
.pv-card { background:linear-gradient(145deg,rgba(30,38,72,.92),rgba(19,25,49,.96)); border:1px solid var(--pv-border); border-radius:18px; padding:18px 20px; margin-bottom:14px; box-shadow:0 10px 28px rgba(0,0,0,.12); }
.pv-muted { color:var(--pv-muted); font-size:.9rem; }
.pv-kicker { color:#cba1f3; font-size:.75rem; text-transform:uppercase; letter-spacing:.13em; font-weight:700; }
.pv-title { font-family:'Manrope',sans-serif; font-size:1.5rem; font-weight:800; color:#fbf9ff; margin:.1rem 0 .35rem 0; }
.pv-section { font-family:'Manrope',sans-serif; font-size:1.08rem; font-weight:700; margin-bottom:10px; }
.pv-pill { display:inline-block; background:#263e4c; color:#a7f3d0; border:1px solid #35665f; border-radius:99px; padding:5px 10px; font-size:.78rem; }
div.stButton > button, div.stDownloadButton > button { border-radius:12px; border:1px solid #5a4d83; background:linear-gradient(110deg,#b66bd1,#7664d9); color:white; font-weight:700; min-height:2.6rem; }
div.stButton > button:hover, div.stDownloadButton > button:hover { border-color:#e2c2ff; color:white; filter:brightness(1.08); }
.stTextInput input, .stTextArea textarea, .stSelectbox div[data-baseweb="select"] { background:#121a35!important; color:#f5f3ff!important; border-color:#414b75!important; border-radius:10px!important; }
[data-testid="stMetric"] { background:#171f3d; padding:13px 16px; border:1px solid #354064; border-radius:14px; }
[data-testid="stMetricLabel"] { color:#c5c9e5!important; }
[data-testid="stMetricValue"] { color:#fff!important; }
hr { border-color:#30395d; }
footer {visibility:hidden;}
</style>
''', unsafe_allow_html=True)

# Demo sequences are clearly labelled teaching examples; users can replace them.
DEMO_FASTA = '''>Human_demo\nATGCTAGCTAGCTACGATCGATCGATGCTAGCTAGCTAACGTTAGCTAGCTAGGCTAACGATCGATCGATGCTAGC\n>Chimp_demo\nATGCTAGCTAGCTACGATCGATCGATGCTAGCTAGCTAACGTTAGCTAGCTAGGCTAACGATCGATCGATGCTAGT\n>Gorilla_demo\nATGCTAGCTAGCTACGATCGATCGATGCTAGCTAGCTAACGTTAGCTAGCTAGGCTAACGATCGATCAATGCTAGT\n>Mouse_demo\nATGATAGCTTGCTACGATCAATCGATGCTAGATAGCTAACGTTAGCTAGTTAGGCTAACGATCAATCGATGCTAGT\n>Rat_demo\nATGATAGCTTGCTACGATCAATCGATGCTAGATAGCTAACGTTAGCTAGTTAGGCTAACGATCAATCGATGCTAGC'''
BASE_COLORS = {'A':'#65bd8b','T':'#ed7e88','G':'#f1bd65','C':'#67b8ee','U':'#b89ce9','-':'#59627f','N':'#8b8fa7'}
CREATOR = 'Anureet Dutta'
CO_CREATORS = ['Sreejeeta Kundu','Aishwarya Banik','Arya Ghosh','Ram Argha Nayak']

# Header with the supplied project logo.
h1, h2 = st.columns([1.5, 3.6], vertical_alignment='center')
with h1:
    if LOGO.exists(): st.image(str(LOGO), width=280)
    else: st.markdown('<div class="pv-title">🧬 PhyloVision</div>', unsafe_allow_html=True)
with h2:
    st.markdown('<div class="pv-kicker">From Sequences to Evolution</div><div style="font-size:1.55rem;font-weight:700;color:#f7f4ff;font-family:Manrope">Explore biological relationships, clearly.</div><div class="pv-muted">A student-friendly workspace for sequence alignment, tree building and interpretation.</div>', unsafe_allow_html=True)
st.markdown('')

with st.sidebar:
    st.markdown('### 🧬 PhyloVision')
    st.caption('BIOINFORMATICS WORKSPACE')
    page = st.radio('Navigate', ['Home','Sequence Input','Alignment Explorer','Phylogenetic Tree','Evolution Lens','Understand My Tree','Sequence Similarity','Creators & About'], label_visibility='collapsed')
    st.markdown('---')
    st.markdown('**Workflow**')
    st.markdown('1. Add sequences\n2. Review alignment\n3. Build a tree\n4. Interpret similarities')
    st.markdown('---')
    st.caption('Educational project • Results depend on input quality and method.')

if 'fasta_text' not in st.session_state: st.session_state.fasta_text = DEMO_FASTA
if 'tree' not in st.session_state: st.session_state.tree = None
if 'aligned_records' not in st.session_state: st.session_state.aligned_records = None
if 'dist_df' not in st.session_state: st.session_state.dist_df = None
if 'tree_method' not in st.session_state: st.session_state.tree_method = 'Neighbor Joining'
if 'sequence_type' not in st.session_state: st.session_state.sequence_type = 'DNA'
if 'bootstrap_support' not in st.session_state: st.session_state.bootstrap_support = {}
if 'tree_signature' not in st.session_state: st.session_state.tree_signature = None


def parse_fasta(text):
    if not text.strip(): raise ValueError('Please paste or upload FASTA sequences first.')
    try: records = list(SeqIO.parse(io.StringIO(text), 'fasta'))
    except Exception as e: raise ValueError(f'Could not read FASTA: {e}')
    if len(records) < 3: raise ValueError('Please provide at least 3 sequences to build a useful example tree.')
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


def pairwise_p_distance(a,b):
    pairs=[(x,y) for x,y in zip(str(a).upper(),str(b).upper()) if x not in '-N?' and y not in '-N?']
    if not pairs: return np.nan
    return sum(x!=y for x,y in pairs)/len(pairs)


def make_distance_matrix(records):
    names=[r.id for r in records]
    mat=np.zeros((len(records),len(records)))
    for i in range(len(records)):
        for j in range(i+1,len(records)):
            d=pairwise_p_distance(records[i].seq,records[j].seq)
            mat[i,j]=mat[j,i]=d
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
        fig.add_trace(go.Scatter(x=[xmap[t]],y=[ymap[t]],mode='markers+text',text=[t.name],textposition='middle right',textfont=dict(color='#eeeaff',size=11),marker=dict(size=7,color=palette[terminals.index(t)%len(palette)]),hovertemplate=f'<b>{t.name}</b><br>Root distance: {xmap[t]:.4f}<extra></extra>',showlegend=False))
    fig.update_layout(height=max(430,n*42),margin=dict(l=10,r=130,t=10,b=10),paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)',xaxis=dict(title='Accumulated branch length (relative evolutionary change)',showgrid=True,gridcolor='rgba(130,145,190,.15)',zeroline=False,color='#c4cbe8',rangemode='tozero'),yaxis=dict(visible=False,autorange='reversed'),font=dict(family='DM Sans',color='#f4f1ff'),showlegend=False)
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
    st.markdown('<div class="pv-card"><div class="pv-kicker">Welcome to your sequence workspace</div><div class="pv-title">From raw sequences to clearer relationships.</div><div class="pv-muted">PhyloVision brings sequence input, alignment inspection, similarity comparison and phylogenetic tree interpretation into one calm, student-friendly workflow.</div></div>',unsafe_allow_html=True)
    a,b,c=st.columns(3)
    for col,icon,title,desc in [(a,'🧬','Alignment Explorer','Inspect matching positions, differences and gaps.'),(b,'🌿','Understand My Tree','Get a plain-language explanation of the inferred tree.'),(c,'▦','Sequence Similarity','Compare pairwise sequence identity in a heatmap and table.')]:
        with col: st.markdown(f'<div class="pv-card" style="min-height:155px"><div style="font-size:1.5rem">{icon}</div><div class="pv-section">{title}</div><div class="pv-muted">{desc}</div></div>',unsafe_allow_html=True)
    st.markdown('<div class="pv-card"><div class="pv-section">Quick start</div><ol><li>Open <b>Sequence Input</b> and upload or paste FASTA records.</li><li>Choose DNA or protein and review any validation messages.</li><li>Open <b>Alignment Explorer</b> to inspect sequence positions.</li><li>Choose <b>Phylogenetic Tree</b> and build the tree.</li><li>Use <b>Understand My Tree</b> and <b>Sequence Similarity</b> to interpret the output.</li></ol><div class="pv-muted">The included sample sequences are illustrative teaching data, not verified reference sequences.</div></div>',unsafe_allow_html=True)

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
        if len({len(r.seq) for r in records}) != 1:
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
                    st.plotly_chart(rectangular_tree_figure(st.session_state.tree),use_container_width=True,config={'displaylogo':False,'scrollZoom':True})
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
    st.markdown('<div class="pv-title">Sequence Similarity</div><div class="pv-muted">Compare pairs of sequences and see how observed identity relates to their differences.</div>',unsafe_allow_html=True)
    try:
        records=current_records(); df=make_distance_matrix(records); sim=(1-df)*100; np.fill_diagonal(sim.values,100)
        fig=go.Figure(data=go.Heatmap(z=sim.values,x=sim.columns,y=sim.index,colorscale=[[0,'#26294b'],[.5,'#8369be'],[1,'#75d4ba']],zmin=0,zmax=100,colorbar=dict(title='Identity %',tickfont=dict(color='#e8e5ff')),hovertemplate='%{y} vs %{x}<br>Identity: %{z:.2f}%<extra></extra>'))
        fig.update_layout(height=max(400,len(records)*44),margin=dict(l=10,r=10,t=15,b=10),paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)',font=dict(color='#f2efff',size=11),xaxis=dict(tickangle=-35),yaxis=dict(autorange='reversed'))
        st.plotly_chart(fig,use_container_width=True,config={'displaylogo':False})
        st.markdown('<div class="pv-card"><div class="pv-section">What does this mean?</div><p>Pairwise identity is the percentage of compared, non-gap positions that match. Higher identity means fewer observed differences across those positions. This is a simple descriptive measure; it is not itself a complete evolutionary model.</p><p class="pv-muted">Gaps and ambiguous characters are excluded from each pairwise comparison. The displayed identity is not a bootstrap confidence score.</p></div>',unsafe_allow_html=True)
        out=sim.round(2).to_csv(index=True)
        st.download_button('Download similarity matrix (CSV)',out,file_name='phylovision_similarity.csv',mime='text/csv')
        st.dataframe(sim.round(2).style.format('{:.2f}%'),use_container_width=True)
    except Exception as e: st.warning(str(e))

else: # Creators & About
    st.markdown('<div class="pv-title">Creators & About</div><div class="pv-muted">The people and purpose behind PhyloVision.</div>',unsafe_allow_html=True)
    st.markdown('<div class="pv-card"><div class="pv-kicker">Project creator</div><div style="font-family:Manrope;font-size:1.5rem;font-weight:800;color:#e1b5ff;margin-top:4px">Anureet Dutta</div><div class="pv-muted">Creator & project lead</div><hr><div class="pv-kicker">Co-creators</div><div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:12px;margin-top:12px">'+''.join(f'<div style="padding:14px 15px;border-radius:12px;background:#202747;border:1px solid #394369"><div style="font-weight:700;color:#f3edff">{name}</div><div class="pv-muted">Co-creator</div></div>' for name in CO_CREATORS)+'</div></div>',unsafe_allow_html=True)
    st.markdown('<div class="pv-card"><div class="pv-section">About the project</div><p>PhyloVision is an educational bioinformatics dashboard designed to connect multiple sequence alignment with phylogenetic tree construction and interpretation. It brings sequence input, an alignment view, a distance-based tree and pairwise identity summaries into one interface.</p><p><b>Learning outcomes:</b> FASTA handling, sequence validation, alignment inspection, pairwise similarity, distance-based tree construction and cautious biological interpretation.</p><p class="pv-muted">Current limitations: sequences must already be aligned and equal length; the tree uses Neighbor Joining with uncorrected p-distance; no bootstrap support is calculated; protein input is accepted for comparison, but this simple distance approach is not a substitute for a validated protein-specific evolutionary model. For research, use suitable validated tools and document sequence accessions and settings.</p></div>',unsafe_allow_html=True)
    st.markdown('<div class="pv-card"><div class="pv-section">Credits & acknowledgement</div><p>Built as a student biotechnology software project, with a focus on making sequence-based evolutionary analysis easier to explore and explain.</p><div class="pv-muted">Better data. Clearer relationships. A deeper understanding of life.</div></div>',unsafe_allow_html=True)

st.markdown('<hr><div style="text-align:center;color:#9ea8cc;font-size:.8rem;padding:8px 0">PhyloVision · From Sequences to Evolution · Educational use</div>',unsafe_allow_html=True)
