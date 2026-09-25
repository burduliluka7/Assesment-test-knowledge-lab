"""Conservative extractive linguistic rules; no domain taxonomy or answer data."""
from dataclasses import dataclass,asdict
import re

MONTH=r'(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)'
DATE=re.compile(r'\b(?:by|before|after|since|until|as of)\s+(?:'+MONTH+r'\s+)?(?:\d{1,2}[, ]+)?(?:19|20)\d{2}\b',re.I)
NEGATIVE=re.compile(r'\b(?:without|excluding|except(?: for)?|neither|(?:does|do|is|are|must|should|can)\s+not)\b',re.I)
BOUNDARY=re.compile(r'\b(?:when|while|with|that|where|provided that|subject to|without|excluding|except(?: for)?|neither|(?:does|do|is|are|must|should|can)\s+not)\b',re.I)
ANSWER=re.compile(r'^\s*(?:which|what)\s+(?:(?:are|is)\s+)?(?:(?:the|a|an|leading|best|available|current|most|suitable)\s+)*(?P<type>methods?|models?|architectures?|datasets?|benchmarks?|articles?|papers?|metrics?|techniques?|tools?|systems?|algorithms?)\b',re.I)

@dataclass
class QuestionDecomposition:
    original_question: str
    answer_type: str|None
    core_task: str|None
    requirements: list
    exclusions: list
    temporal_condition: str|None
    source_spans: dict
    components: list
    fallback: bool
    warnings: list

    def to_dict(self): return asdict(self)

def decompose_question(question,declared_type='A'):
    """Only contiguous original substrings become fields; offsets are end-exclusive."""
    spans={}; masked=list(question); dates=list(DATE.finditer(question))
    for m in dates:
        for i in range(m.start(),m.end()): masked[i]=' '
    visible=''.join(masked)
    spans['temporal_condition']=[dict(text=m.group(),source_span=m.group(),start=m.start(),end=m.end()) for m in dates]
    def span(start,end):
        while start<end and (question[start].isspace() or question[start] in '(),;:?'): start+=1
        while end>start and (question[end-1].isspace() or question[end-1] in ',;:?.'): end-=1
        return dict(text=question[start:end],source_span=question[start:end],start=start,end=end)
    answer=ANSWER.search(visible); answer_type=None; start=0
    if answer is None and re.match(r'^\s*(?:what|which)\b',visible,re.I):
        head=re.split(r'\b(?:for|when|while|with|without|that)\b',visible,1,flags=re.I)[0]
        matches=list(re.finditer(r'\b(?P<type>methods?|models?|architectures?|datasets?|benchmarks?|articles?|papers?|metrics?|techniques?|tools?|systems?|algorithms?)\b',head,re.I))
        answer=matches[-1] if matches else None
    if answer:
        name=answer.group('type').lower(); answer_type=name[:-1] if name.endswith('s') else name
        if answer_type=='metrice': answer_type='metric'
        spans['answer_type']=span(answer.start('type'),answer.end('type')); start=answer.end()
    # Generic interrogative/copula framing, not scientific vocabulary.
    # Coordinate answer heads do not become task requirements.
    coordinate=re.match(r'\s+(?:and|or)\s+(?:datasets?|benchmarks?|methods?|models?|articles?|metrics?|techniques?)\b',visible[start:],re.I)
    if coordinate:
        answer_type=None; start+=coordinate.end()
    prefix=re.match(r'[\s(),]*(?:(?:are|is)(?!\s+not)[\s(),]+)?(?:(?:best|most|commonly|frequently|typically)[\s(),]+)?(?:(?:suited|suitable|used|useful|designed|appropriate)[\s(),]+)?(?:for\s+|to\s+)?',visible[start:],re.I)
    if prefix: start+=prefix.end()
    # Extract temporal gaps without moving original offsets.
    segments=[]; last=start
    for m in dates:
        if m.end()<=start: continue
        if last<m.start(): segments.append((last,m.start()))
        last=max(last,m.end())
    if last<len(question): segments.append((last,len(question)))
    components=[]; requirements=[]; exclusions=[]; core=None
    for a,b in segments:
        boundaries=list(BOUNDARY.finditer(question,a,b)); cursor=a; mode='core'; connector=''
        chunks=[]
        for m in boundaries:
            if m.group().lower()=='with' and re.search(r'\b(?:dealing|working|associated|consistent|compatible)\s*$',question[cursor:m.start()],re.I): continue
            if m.start()>cursor: chunks.append((cursor,m.start(),mode,connector))
            mode='exclusion' if NEGATIVE.fullmatch(m.group()) else 'requirement'; connector=m.group().lower(); cursor=m.end()
        if cursor<b: chunks.append((cursor,b,mode,connector))
        for left,right,kind,joiner in chunks:
            # Coordinated explicit requirements are separate; preserve OR as alternatives.
            pattern=r'\s*;\s*' if kind=='core' else r'\s+(?:and|nor|but|for)\s+|\s*;\s*'
            splits=list(re.finditer(pattern,question[left:right],re.I))
            splits=[m for m in splits if not (m.group().strip().lower()=='for' and re.search(r'\b(?:support|capacity|need|requirement|ability)\s*$',question[left:left+m.start()],re.I))]
            parts=[]; p=left
            for m in splits: parts.append((p,left+m.start())); p=left+m.end()
            parts.append((p,right))
            for x,y in parts:
                item=span(x,y)
                if not item['text'] or not re.search(r'\w',item['text']): continue
                # Leading balancing is linguistic scaffolding, not a requirement itself.
                lead=re.match(r'balancing\s+',item['text'],re.I)
                if lead: item=span(item['start']+lead.end(),item['end'])
                actual='core_task' if core is None and kind=='core' else 'exclusion' if kind=='exclusion' else 'requirement'
                if actual=='core_task': core=item['text']; spans['core_task']=item
                elif actual=='exclusion': exclusions.append(item['text']); spans.setdefault('exclusions',[]).append(item)
                else: requirements.append(item['text']); spans.setdefault('requirements',[]).append(item)
                components.append(dict(item,kind=actual,connector=joiner))
    fallback=core is None
    if fallback:
        # Use non-temporal text only; original is always separately retained.
        choices=[span(a,b) for a,b in segments if re.search(r'\w',visible[a:b]) and not NEGATIVE.search(visible[a:b])]
        if not choices and spans.get('answer_type'): choices=[spans['answer_type']]
        item=max(choices,key=lambda s:len(s['text']),default=dict(text='',source_span='',start=0,end=0))
        core=item['text'] or None; spans['core_task']=item
        if core: components.insert(0,dict(item,kind='core_task',connector=''))
    for i,c in enumerate(components): c['id']=f'component_{i}'
    warnings=[]
    if fallback: warnings.append('No confident core-task boundary; extractive fallback')
    if any(re.search(r'\bor\b',c['text'],re.I) for c in components): warnings.append('Alternative clause retained intact; no Boolean parsing')
    return QuestionDecomposition(question,answer_type,core,requirements,exclusions,'; '.join(m.group() for m in dates) or None,spans,components,fallback,warnings)

def candidate_hypothesis(name,decomposition,component):
    """Generic literal templates; no paraphrased scientific facts are introduced."""
    core=decomposition.core_task or 'the requested task'; text=component['text']
    if component['kind']=='core_task': return f'{name} is suitable for {text}.'
    if component['kind']=='exclusion': return f'{name} involves {text} for {core}.'  # violation, not desired support
    if component['connector'] in {'when','where','while'}: return f'{name} is suitable for {core} when {text}.'
    return f'{name} supports {core} with {text}.'
