"""Publication visibility; no benchmark input or origin-year filtering."""


def snapshot(graph, year):
    nodes = [n for n in graph["nodes"] if n["first_seen_year"] <= year]
    ids = {n["id"] for n in nodes}
    edges = [
        e
        for e in graph["hyperedges"]
        if e["year"] <= year
        and (e["provenance"].get("article_year") or e["year"]) <= year
        and set(e["members"]) <= ids
    ]
    return dict(snapshot=year, nodes=nodes, hyperedges=edges)


def anomalies(graph):
    nodes = {n["id"]: n for n in graph["nodes"]}
    rows = []
    for e in graph["hyperedges"]:
        latest = max(nodes[v]["first_seen_year"] for v in e["members"])
        article = e["provenance"].get("article_year")
        flags = []
        if e["year"] < latest:
            flags.append("before_member_visibility")
        if article and e["year"] < article:
            flags.append("before_asserting_article")
        if flags:
            rows.append(
                dict(
                    edge_id=e["id"],
                    edge_year=e["year"],
                    latest_member_year=latest,
                    article_year=article,
                    visible_from=max(e["year"], latest, article or e["year"]),
                    flags=flags,
                    members=e["members"],
                )
            )
    return dict(
        counts={
            f: sum(f in r["flags"] for r in rows)
            for f in ["before_member_visibility", "before_asserting_article"]
        },
        edges=rows,
        origin_after_first_seen=[
            n["id"]
            for n in nodes.values()
            if n.get("origin_year") and n["origin_year"] > n["first_seen_year"]
        ],
    )
