"""Evaluation-only target resolution. No answer-key data enters search routing."""
import re
import unicodedata

ENTITY_TYPES = frozenset({'method','technique','component','dataset','task','metric','problem','future_topic','article','cited_work'})
ALL_TYPES = ENTITY_TYPES | {'claim','author','cited_work'}
ROUTES = {
    'methods': frozenset({'method','technique','component','cited_work'}),
    'data': frozenset({'dataset','task','metric','article','cited_work'}),
    'challenges': frozenset({'problem','future_topic','claim','task'}),
    'mechanisms': frozenset({'method','technique','component','claim'}),
    'all': ALL_TYPES,
}

def normalize(text):
    text=unicodedata.normalize('NFKC',str(text)).casefold()
    text=''.join('-' if unicodedata.category(c)=='Pd' else c for c in text)
    # Keep symmetry groups, signs and versions. E(3), E3 and E(2) are distinct.
    return re.sub(r'[^\w+()\-]+',' ',text).strip()

def contains_name(text,name):
    return bool(re.search(r'(?<![\w+\-])'+re.escape(normalize(name))+r'(?![\w+\-])',normalize(text)))

def route_question(question,declared_type='A'):
    """Generic language rules only; never accepts question IDs or target names."""
    q=question.casefold(); head=q[:100]
    if re.search(r'\b(?:methods?|architectures?|models?)\s+(?:and|or)\s+(?:datasets?|benchmarks?)\b|\b(?:datasets?|benchmarks?)\s+(?:and|or)\s+(?:methods?|architectures?|models?)\b',head):
        name='all'; reason='mixed entity families: all-type fallback'
    elif re.search(r'\b(?:what|which)\s+(?:\w+\s+){0,3}(?:datasets?|benchmarks?)\b',head):
        name='data'; reason='interrogative asks for datasets or benchmarks'
    elif re.search(r'\b(?:open|scientific)\s+challenges?\b',q):
        name='challenges'; reason='question asks about scientific challenges'
    elif declared_type=='B' or re.search(r'^how\b|\bmechanisms?\b|\bevidences?\b',q):
        name='mechanisms'; reason='explanatory/evidence question or declared Type B'
    elif re.search(r'\b(?:methods?|architectures?|models?)\b',head):
        name='methods'; reason='interrogative names methods, architectures or models'
    else:
        name='all'; reason='weak routing signal: all-type fallback'
    return dict(profile=name,types=sorted(ROUTES[name]),rule=reason,confidence='fallback' if name=='all' else 'rule_based')

def resolve_target(target,nodes):
    candidates=[n for n in nodes if n['type'] in ENTITY_TYPES or n['type']=='claim']
    norm=normalize(target)
    exact=[n for n in candidates if normalize(n['surface_form'])==norm]
    # Containment is limited to entity-like nodes, never assertions mentioning a name.
    aliases=[n for n in candidates if n['type'] in {'method','technique','component','dataset'} and contains_name(n['surface_form'],target)]
    chosen=exact or aliases
    distinct={normalize(n['surface_form']) for n in chosen}
    if exact: status='exact'; accepted=chosen
    elif len(distinct)==1 and chosen: status='safe_alias'; accepted=chosen
    elif chosen: status='ambiguous'; accepted=[]
    else: status='unmapped'; accepted=[]
    return dict(supplied_target=target,status=status,mapping_rule='exact_normalized' if exact else 'whole_identifier_containment' if chosen else 'no_lexical_match',
        confidence='lexical_proxy_not_human_verified' if accepted else 'unresolved',
        ambiguity_flag=len(chosen)>1,node_ids=[n['id'] for n in accepted],
        candidates=[dict(node_id=n['id'],node_type=n['type'],surface_form=n['surface_form']) for n in chosen],
        assumption='Identical normalized surface mentions count as one named target; distinct containing forms remain ambiguous')

def map_benchmark(gt,nodes,cutoff):
    rows=[]
    for qid,item in gt.items():
        categories={'expected_targets':item.get('expected_methods',[]),**item.get('fp_categories',{})}
        for category,names in categories.items():
            for name in names:
                rows.append(dict(question_id=qid,question_type=item['type'],category=category,cutoff=cutoff,**resolve_target(name,nodes)))
    return rows

def mapping_summary(rows):
    expected=[r for r in rows if r['category']=='expected_targets']; unique={r['supplied_target']:r for r in expected}
    def stats(rr):
        return dict(total=len(rr),mapped=sum(bool(r['node_ids']) for r in rr),
            exact=sum(r['status']=='exact' for r in rr),safe_alias=sum(r['status']=='safe_alias' for r in rr),
            ambiguous=sum(r['status']=='ambiguous' for r in rr),unmapped=sum(r['status']=='unmapped' for r in rr))
    by_type={}
    for r in expected:
        for typ in {c['node_type'] for c in r['candidates'] if c['node_id'] in r['node_ids']}:
            by_type[typ]=by_type.get(typ,0)+1
    return dict(unique=stats(list(unique.values())),question_target_instances=stats(expected),mapped_instances_by_node_type=by_type)

# The legacy resolver above remains unchanged for reproducible v2 baselines.
def singular_name(text):
    """Conservative final-token plural variant, never strip version/sign tokens."""
    value=normalize(text)
    return re.sub(r'\b([a-z]{3,})s$',lambda m:m[1] if not m[0].endswith(('ss','us','is')) else m[0],value)

class TargetResolver:
    """Graph-only identity evidence; benchmark claims/answers never enter its index."""
    def __init__(self,graph):
        self.graph=graph; self.nodes=graph['nodes']; self.lookup={n['id']:n for n in self.nodes}
        self.entities=[n for n in self.nodes if n['type'] in ENTITY_TYPES]
        self.by_name={}; self.article_nodes={}
        for n in self.nodes:
            if n['type'] in ENTITY_TYPES: self.by_name.setdefault(normalize(n['surface_form']),[]).append(n)
            for a in n.get('provenance',{}).get('articles',[]): self.article_nodes.setdefault(a,[]).append(n)

    def resolve(self,label,mode='generic',_depth=0):
        norm=normalize(label); exact=self.by_name.get(norm,[])
        base=dict(label=label,resolution_type='UNRESOLVED',resolved_node_ids=[],node_types=[],component_resolutions=[],
            confidence='unresolved',signals=[],reason='No safe identity or repeated literal evidence found',canonical=False,representable=False,evidence_node_ids=[],candidates=[])
        def accepted(nodes,kind,signal):
            context=[n['id'] for n in self.nodes if n['type'] in {'claim','article'} and contains_name(n['surface_form'],label)][:8]
            return dict(base,resolution_type=kind,resolved_node_ids=sorted(n['id'] for n in nodes),node_types=sorted({n['type'] for n in nodes}),
                canonical=True,representable=True,confidence='lexical_identity_proxy',signals=[signal],reason=signal,evidence_node_ids=context,
                candidates=[dict(node_id=n['id'],surface_form=n['surface_form'],node_type=n['type']) for n in nodes])
        if exact: return accepted(exact,'EXACT_NODE','Identical normalized scientific name; all node mentions retained')
        if mode=='exact': return base
        # Exact first protects scientific names containing + or /; terminal ++ is never split.
        parts=[p.strip() for p in re.split(r'(?<=[\w)])\s*(?:→|->|(?<!\+)\+(?!\+)|/)\s*(?=[\w(])|\s+(?:combined with|with|and)\s+',label,flags=re.I) if p.strip()]
        if len(parts)>1 and _depth<2:
            components=[self.resolve(p,mode,_depth+1) for p in parts]
            sets=[{a for x in c['resolved_node_ids'] for a in self.lookup[x].get('provenance',{}).get('articles',[])} for c in components]
            common=set.intersection(*sets) if sets else set()
            complete=all(c['canonical'] for c in components)
            return dict(base,resolution_type='COMPOSITE',component_resolutions=components,confidence='graph_supported_components' if complete and common else 'incomplete_or_unverified_combination',
                representable=bool(complete and common),signals=['syntactic_composition'],component_article_overlap=sorted(common),
                reason='Components resolved independently; representable only when all canonical components share article provenance; no synthetic graph node created')
        plural=[n for n in self.entities if singular_name(n['surface_form'])==singular_name(label)]
        if plural: return accepted(plural,'ALIAS_NODE','Only a conservative final-token singular/plural difference')
        defined=[n for n in self.entities if re.search(r'\('+re.escape(label)+r'\)',n['surface_form'],re.I)] if re.fullmatch(r'[A-Z][A-Z0-9-]{1,9}',label) else []
        identifier=bool(re.search(r'[a-z][A-Z]|\d|[-()]',label) or re.fullmatch(r'[A-Z]{3,}',label))
        contained=[n for n in self.entities if n['type'] in {'method','technique','component','dataset'} and contains_name(n['surface_form'],label)] if identifier else []
        aliases=defined or contained; distinct={normalize(n['surface_form']) for n in aliases}
        if len(distinct)==1:
            return accepted(aliases,'ALIAS_NODE','Explicit parenthetical acronym definition' if defined else 'Unique whole-identifier containment in entity name')
        if aliases:
            return dict(base,resolution_type='AMBIGUOUS',confidence='multiple_identity_candidates',reason='Multiple distinct names share the identifier; none forced',
                candidates=[dict(node_id=n['id'],surface_form=n['surface_form'],node_type=n['type']) for n in aliases])
        evidence=[n for n in self.nodes if n['type'] in {'claim','article','cited_work','future_topic'} and contains_name(n['surface_form'],label)]
        if len({normalize(n['surface_form']) for n in evidence})>=2:
            return dict(base,resolution_type='EVIDENCE_BACKED',evidence_node_ids=sorted(n['id'] for n in evidence),representable=True,
                confidence='repeated_literal_evidence_not_entity_identity',signals=['repeated_whole_phrase_in_graph'],reason='Represented by multiple literal evidence texts; no canonical entity identity asserted')
        return dict(base,evidence_node_ids=[n['id'] for n in evidence],reason='No canonical name; insufficient repeated evidence (semantic relatedness alone cannot certify identity)')
