"""Cooperative Python file/import firewall; not hostile native-code containment."""
import os
import sys
from pathlib import Path

FORBIDDEN = ('ground_truth','target_annotations','claim_annotations','strict_target','target_mapping',
             'metrics','failure','oracle','per_question_evaluation','positive_controls')


def install(root, prediction=True, frozen_files=()):
    root=Path(root).resolve();repo=root.parent
    frozen={Path(p).resolve() for p in frozen_files}
    accesses=[];created=set()

    def forbidden(path):
        relative=path.relative_to(root).as_posix().lower()
        if any(x in relative for x in FORBIDDEN):return True
        if relative.startswith(('report/','notes/','archive/','artifacts/evaluation/')):return True
        if relative.startswith('src/') and ('evaluation' in path.name or path.name=='strict_targets.py'):return True
        return False

    def allowed_read(path):
        if path in created:return True
        relative=path.relative_to(root).as_posix()
        if relative in ('config.json','model_manifest.json','data/questions.csv','data/tkh_collection10.json'):return True
        if relative.startswith('src/') and path.suffix in ('.py','.pyc'):return True
        if relative=='scripts/predict.py':return True
        # Existing caches are content-addressed, prediction-independent model inputs/outputs.
        return relative.startswith(('.cache/models/','.cache/embeddings/','.cache/requirement_inference/',
                                    '.cache/hf_home/','.cache/torch_home/','.cache/tmp/','.cache/temp/'))

    def deny(path, reason):
        if path.is_relative_to(repo):accesses.append(dict(path=path.relative_to(repo).as_posix(),allowed=False,reason=reason))
        raise PermissionError(reason+': '+str(path))

    def mutation(path):
        if not path.is_relative_to(root):deny(path,'Writes restricted to V2')
        if any(path==p or path in p.parents for p in frozen):deny(path,'Frozen artifact is read-only')

    def guard(event,args):
        if event=='import' and prediction:
            name=args[0]
            if (name.startswith('tkh_abstraction') or '.' not in name) and any(
                x in ('benchmark','evaluation','evaluate','strict_targets') or x.endswith('_evaluation') for x in name.split('.')):
                raise PermissionError('Evaluation imports forbidden before prediction freeze: '+name)
        if event=='open':
            raw,mode,flags=args
            if not isinstance(raw,(str,bytes,os.PathLike)):return
            path=Path(os.fsdecode(raw)).resolve()
            writing=(isinstance(mode,str) and any(x in mode for x in 'wax+')) or bool((flags or 0)&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND))
            if writing:
                mutation(path)
                if prediction and path.is_relative_to(root) and forbidden(path):deny(path,'Forbidden prediction output category')
                created.add(path)
                return
            if prediction:
                if path.is_relative_to(root):
                    if forbidden(path) or not allowed_read(path):deny(path,'Non-input or historical/evaluation read forbidden')
                elif path.is_relative_to(repo):
                    if not any(x in path.parts for x in ('site-packages','.venv','.repro-venv')):
                        deny(path,'Prediction requires copied V2 inputs')
                elif any(x in path.name.lower() for x in FORBIDDEN):deny(path,'Answer-data read forbidden')
            if path.is_relative_to(repo):accesses.append(dict(path=path.relative_to(repo).as_posix(),allowed=True))
        if event in ('os.remove','os.rmdir','os.mkdir','os.rename','os.link','os.symlink'):
            count=2 if event in ('os.rename','os.link','os.symlink') else 1
            for raw in args[:count]:
                if isinstance(raw,(str,bytes,os.PathLike)):mutation(Path(os.fsdecode(raw)).resolve())
        if event in ('os.chmod','os.utime','os.truncate') and isinstance(args[0],(str,bytes,os.PathLike)):
            mutation(Path(os.fsdecode(args[0])).resolve())
    sys.addaudithook(guard)
    return accesses
