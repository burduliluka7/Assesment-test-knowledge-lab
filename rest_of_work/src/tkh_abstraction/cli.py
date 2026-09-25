import argparse
import sys
from .config import read_config
from .pipeline import run

def main(argv=None):
    p=argparse.ArgumentParser(description='Reproduce the temporal hypergraph assessment')
    p.add_argument('command',choices=['validate-data','describe','build','label','evaluate','evaluate-v2','evaluate-context','evaluate-diffusion','evaluate-reranking','evaluate-constraints','evaluate-conditioned','evaluate-hierarchy-retrieval','report','run-all'])
    p.add_argument('--config',default='configs/default.yaml'); p.add_argument('--snapshot',type=int,action='append')
    p.add_argument('--variant',choices=['semantic','structural','static','temporal','pairwise'])
    p.add_argument('--seed',type=int); p.add_argument('--fast',action='store_true')
    p.add_argument('--stage',action='append',choices=['R0','R1','R1_direct','R2','R3','R4','R5'])
    p.add_argument('--cutoff',type=int,action='append')
    p.add_argument('--resolver',choices=['exact','generic'])
    p.add_argument('--candidate-k',type=int,choices=[50,100])
    p.add_argument('--diffusion-stage',action='append',choices=['R0','R1','R3','H0','H0_shuffled','H0_pairwise','H1','H2','H3'])
    p.add_argument('--no-sensitivity',action='store_true')
    p.add_argument('--questions'); p.add_argument('--ground-truth')
    args=p.parse_args(argv)
    try:
        cfg=read_config(args.config)
        if args.fast:
            cfg.update(output='artifacts/fast',snapshots=[2020,2026],variants=['temporal'],null_permutations=10,sensitivity=False)
        if args.snapshot: cfg['snapshots']=sorted(set(args.snapshot))
        if args.variant: cfg['variants']=[args.variant]
        if args.seed is not None: cfg['seed']=args.seed
        if args.command=='evaluate-hierarchy-retrieval':
            if args.cutoff: cfg['final_cutoffs']=args.cutoff
            from .final_pipeline import run_final
            run_final(cfg,args.questions,args.ground_truth)
        elif args.command=='evaluate-conditioned':
            if args.cutoff: cfg['conditioned_cutoffs']=args.cutoff
            from .conditioned_pipeline import run_conditioned
            run_conditioned(cfg,args.questions,args.ground_truth)
        elif args.command=='evaluate-constraints':
            if args.cutoff: cfg['constraint_cutoffs']=args.cutoff
            from .constraint_pipeline import run_constraints
            run_constraints(cfg,args.questions,args.ground_truth)
        elif args.command=='evaluate-reranking':
            if args.cutoff: cfg['reranker_cutoffs']=args.cutoff
            from .reranker_pipeline import run_reranking
            run_reranking(cfg,args.questions,args.ground_truth)
        elif args.command=='evaluate-diffusion':
            if args.cutoff: cfg['diffusion_cutoffs']=args.cutoff
            from .diffusion_pipeline import run_diffusion
            run_diffusion(cfg,args.diffusion_stage,not args.no_sensitivity)
        elif args.command=='evaluate-context':
            if args.cutoff: cfg['context_cutoffs']=args.cutoff
            if args.resolver: cfg['context_resolver']=args.resolver
            if args.candidate_k: cfg['context_candidate_k']=args.candidate_k
            from .v2_pipeline import run_context
            run_context(cfg,args.stage)
        elif args.command=='evaluate-v2':
            from .v2_pipeline import run_v2
            run_v2(cfg)
        else: run(cfg,args.command)
    except (ValueError,FileNotFoundError,ImportError,OSError) as e:
        print(f'tkh: {e}',file=sys.stderr); return 2
    return 0

if __name__=='__main__': raise SystemExit(main())
