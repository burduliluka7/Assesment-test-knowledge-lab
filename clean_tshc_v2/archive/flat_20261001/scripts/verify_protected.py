"""Compare every protected file and complete manifest bytes; no protected writes."""
import argparse
import hashlib
import json
import tempfile
from pathlib import Path
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
PROTECTED = ROOT.parent / 'clean_tshc'


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--before', type=Path, default=Path(tempfile.gettempdir()) / 'clean_tshc_before_v2.sha256')
    parser.add_argument('--initialize', action='store_true')
    args = parser.parse_args()
    actual = {p.relative_to(PROTECTED).as_posix(): p for p in PROTECTED.rglob('*') if p.is_file()}
    if args.initialize:
        if args.before.exists():
            raise FileExistsError('Never overwrite an existing before manifest; choose a fresh --before path')
        data = ''.join(digest(actual[name]) + '  ' + name + '\n' for name in sorted(actual)).encode('utf8')
        args.before.parent.mkdir(parents=True, exist_ok=True)
        args.before.write_bytes(data)
        print('Before manifest initialized:', args.before, len(actual), 'files')
        return
    before = args.before.read_bytes()
    expected = {line.split('  ', 1)[1]: line.split('  ', 1)[0] for line in before.decode('utf8').splitlines()}
    # Retain baseline ordering and newline style so comparison is byte-for-byte
    # even when the original manifest used PowerShell's culture-aware sort.
    newline = '\r\n' if b'\r\n' in before else '\n'
    names = [name for name in expected if name in actual] + sorted(set(actual) - set(expected))
    hashes = {name: digest(actual[name]) for name in names}
    after = ''.join(hashes[name] + '  ' + name + newline for name in names).encode('utf8')
    out = ROOT / 'outputs/audit'
    out.mkdir(parents=True, exist_ok=True)
    (out / 'clean_tshc_before_v2.sha256').write_bytes(before)
    (out / 'clean_tshc_after_v2.sha256').write_bytes(after)
    result = dict(unchanged=before == after, files=len(actual), before=str(args.before),
        before_manifest_sha256=hashlib.sha256(before).hexdigest(), after_manifest_sha256=hashlib.sha256(after).hexdigest(),
        added=sorted(set(actual)-set(expected)), removed=sorted(set(expected)-set(actual)),
        changed=[name for name in expected if name in hashes and hashes[name] != expected[name]],
        comparison='Every file content plus exact complete manifest bytes; preserves original manifest ordering',
        checked_at=datetime.now(timezone.utc).isoformat())
    (out / 'protected_verification.json').write_text(json.dumps(result, indent=2), encoding='utf8')
    print('CLEAN_TSHC UNCHANGED:', 'YES' if result['unchanged'] else 'NO')
    if not result['unchanged']:
        raise SystemExit('Protected reference changed: stop and investigate before finishing')


if __name__ == '__main__':
    main()
