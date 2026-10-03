"""Fail-closed Python audit hooks. Not an OS security boundary against hostile native code."""
import os
import sys
from pathlib import Path

FORBIDDEN = {'ground_truth.json', 'target_annotations.json', 'claim_annotations.json'}


def install(root, prediction=True):
    root = Path(root).resolve()
    repo = root.parent
    protected = [(repo / n).resolve() for n in ('clean_tshc', 'rest_of_work')]
    accesses = []

    def guard(event, args):
        if event == 'import' and prediction:
            name = args[0]
            if (name.startswith('tkh_abstraction') or '.' not in name) and any(x in ('benchmark', 'evaluation', 'evaluate', 'strict_targets') or x.endswith('_evaluation') for x in name.split('.')):
                raise PermissionError('Evaluation imports forbidden before prediction freeze: ' + name)
        if event == 'open':
            raw, mode, flags = args
            if not isinstance(raw, (str, bytes, os.PathLike)):
                return
            path = Path(os.fsdecode(raw)).resolve()
            writing = (isinstance(mode, str) and any(x in mode for x in 'wax+')) or bool((flags or 0) & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND))
            if writing and not path.is_relative_to(root):
                raise PermissionError('Writes restricted to V2: ' + str(path))
            if prediction and path.name.lower() in FORBIDDEN:
                raise PermissionError('Answer data forbidden before prediction freeze: ' + str(path))
            if prediction and not writing and (any(path.is_relative_to(root / p) for p in ('archive', 'report', 'notes')) or path.name in {'metrics.json', 'per_question_evaluation.json', 'strict_target_support.json', 'positive_controls.json'}):
                raise PermissionError('Historical/evaluation artifacts forbidden during prediction: ' + str(path))
            if prediction and any(path.is_relative_to(p) for p in protected):
                # Third-party dependencies in a preexisting interpreter are allowed read-only.
                if 'site-packages' not in path.parts and not any(x in path.parts for x in ('.venv', '.repro-venv')):
                    raise PermissionError('Prediction must use copied V2 inputs: ' + str(path))
            if path.is_relative_to(root / 'data') and not writing:
                accesses.append(str(path))
        if event in ('os.remove', 'os.rmdir', 'os.mkdir', 'os.rename'):
            count = 2 if event == 'os.rename' else 1
            for raw in args[:count]:
                if isinstance(raw, (str, bytes, os.PathLike)):
                    path = Path(os.fsdecode(raw)).resolve()
                    if not path.is_relative_to(root):
                        raise PermissionError('Filesystem mutation outside V2: ' + str(path))
    sys.addaudithook(guard)
    return accesses
