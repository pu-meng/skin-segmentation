"""Mirror the pre-existing run's state, then retire its two output path aliases."""
import argparse
import json
import os
import time
from pathlib import Path


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--old-runtime',type=Path,required=True)
    args=parser.parse_args()
    project=Path(__file__).resolve().parents[1]
    destination=project/'log/Conv_Attention_Mamba_整网对比/training_status.json'
    cleanup=project/'references/整理记录/cleanup_20261026.json'
    while True:
        data=json.loads((args.old_runtime/'training_status.json').read_text())
        data['runtime_migration']={'source_status':str(args.old_runtime/'training_status.json'),
            'current_result_root':str(project/'log/Conv_Attention_Mamba_整网对比/runs'),
            'training_restarted':False}
        temporary=destination.with_suffix('.migration.tmp')
        temporary.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n');os.replace(temporary,destination)
        if data.get('status')=='training_validation_complete_features_pending':
            for kind in ('conv','transformer','mamba'):
                folder=project/'log/Conv_Attention_Mamba_整网对比/runs'/kind/'seed42'
                if len(json.loads((folder/'history.json').read_text()))!=300 or not (folder/'final.pth').exists():
                    raise RuntimeError('Final artifacts incomplete; original aliases retained')
            removed=[]
            for name in ('runs','training_logs'):
                alias=args.old_runtime/name
                if alias.is_symlink():alias.unlink();removed.append(str(alias))
            record=json.loads(cleanup.read_text());record['active_legacy_aliases']=[]
            record['retired_after_training']=removed;record['running_job_cleanup']='complete'
            cleanup.write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n')
            return
        if data.get('status') in ('one_or_more_runs_failed','failed'):
            raise RuntimeError('Original run failed; preserve its paths for inspection')
        time.sleep(15)


if __name__=='__main__':main()
