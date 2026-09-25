"""Answer-key oracle diagnostics. Never imported by candidate generation or scoring."""
def oracle_ceiling(metric,candidates,candidate_k,k=10):
    units=metric['canonical_evaluation_units']; masks=set()
    for row in candidates[:candidate_k]:
        nodes=set(row['node_ids'])
        mask=sum(1<<i for i,u in enumerate(units) if nodes&set(u['node_ids']))
        if mask: masks.add(mask)
    # Exact set coverage, retaining minimum slots for each covered subset.
    # This also supports an entity group covering more than one relevance unit.
    states={0:0}
    for mask in sorted(masks):
        for prior,slots in list(states.items()):
            merged=prior|mask
            if slots<k and slots+1<states.get(merged,k+1): states[merged]=slots+1
    best=max(states,key=lambda m:(m.bit_count(),-states[m],-m))
    present=0
    for mask in masks: present|=mask
    found=best.bit_count(); total=len(units)
    raw=sum(len(u['supplied_labels']) for i,u in enumerate(units) if best&(1<<i))
    return dict(candidate_k=candidate_k,canonical_targets=total,canonical_candidates=present.bit_count(),oracle_found_at10=found,
        oracle_recall_at10=found/total if total else None,raw_recovered_lower_bound=raw/metric['total_expected'] if metric['total_expected'] else None,
        oracle_raw_found=raw,oracle_slots=states[best],
        interpretation='Evaluation-only exact maximum canonical coverage within ten candidate slots; aliases merged before coverage')
