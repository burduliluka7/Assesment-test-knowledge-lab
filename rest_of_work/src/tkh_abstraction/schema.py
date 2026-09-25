from collections import Counter

def valid_year(y):
    return isinstance(y,int) and not isinstance(y,bool) and 1900<=y<=2100

def validate_data(data):
    issues=[]
    def note(code, ident, severity="warning", **detail):
        issues.append(dict(code=code,id=ident,severity=severity,**detail))
    semantics=data.get('meta',{}).get('date_semantics',{})
    if not semantics:
        note('missing_date_semantics','meta','error')
    nodes=data.get('nodes',[]); edges=data.get('hyperedges',[])
    for kind,rows in [('node',nodes),('edge',edges)]:
        for ident,n in Counter(r.get('id') for r in rows).items():
            if n>1 or not ident: note('duplicate_or_missing_id',ident,'error',kind=kind,count=n)
    lookup={n['id']:n for n in nodes if n.get('id')}
    for n in nodes:
        if not n.get('type'): note('missing_node_type',n.get('id'),'error')
        if not n.get('surface_form','').strip(): note('missing_surface_form',n.get('id'),'error')
        for field in ['year','first_seen_year','last_seen_year']:
            if not valid_year(n.get(field)): note('missing_or_malformed_date',n['id'],'error',field=field)
        if n.get('origin_year') is not None and not valid_year(n['origin_year']): note('malformed_origin',n['id'],'error')
        if valid_year(n.get('first_seen_year')) and valid_year(n.get('last_seen_year')) and n['first_seen_year']>n['last_seen_year']:
            note('reversed_seen_dates',n['id'],'error')
        if valid_year(n.get('origin_year')) and valid_year(n.get('first_seen_year')) and n['origin_year']>n['first_seen_year']:
            note('origin_after_first_seen',n['id'])
    incident=set()
    for e in edges:
        if not e.get('relation_type'): note('missing_relation_type',e.get('id'),'error')
        if not valid_year(e.get('year')): note('missing_or_malformed_date',e['id'],'error',field='year')
        members=e.get('members',[]); incident.update(members)
        if len(members)<2 or len(set(members))!=len(members): note('invalid_members',e['id'],'error')
        missing=set(members)-lookup.keys()
        if missing: note('dangling_members',e['id'],'error',members=sorted(missing))
        py=e.get('provenance',{}).get('article_year')
        if py is not None and not valid_year(py): note('malformed_provenance_year',e['id'],'error')
        if valid_year(e.get('year')):
            if valid_year(py) and py>e['year']: note('edge_before_asserting_paper',e['id'],edge_year=e['year'],article_year=py)
            future=[x for x in members if x in lookup and valid_year(lookup[x].get('first_seen_year')) and lookup[x]['first_seen_year']>e['year']]
            if future: note('edge_before_members_seen',e['id'],members=future)
    isolated=sorted(lookup.keys()-incident)
    return dict(status='invalid' if any(i['severity']=='error' for i in issues) else 'valid_with_warnings' if issues else 'valid',
        nodes=len(nodes),hyperedges=len(edges),date_semantics=semantics,issues=issues,
        issue_counts=dict(Counter(i['code'] for i in issues)),isolated_nodes=isolated,
        node_types=dict(Counter(n.get('type') for n in nodes)),relation_types=dict(Counter(e.get('relation_type') for e in edges)),
        metadata_count_comparison={key:dict(declared=data.get('meta',{}).get(key),measured=measured) for key,measured in [('nodes',len(nodes)),('hyperedges',len(edges))]})
