#!/usr/bin/env python3
"""
Pure evaluation-only script - skips all inference and model loading.
Only loads existing predictions from xlsx files and evaluates them.
"""

import argparse
import json
import os
import os.path as osp
from vlmeval.config import supported_VLM
from vlmeval.dataset import build_dataset
from vlmeval.smp import *

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate existing prediction files without running inference."
    )
    parser.add_argument('--data', type=str, nargs='+', required=True, help='Dataset names')
    parser.add_argument('--model', type=str, required=True, help='Model name')
    parser.add_argument('--work-dir', type=str, default='./outputs', help='Results directory')
    parser.add_argument('--judge', type=str, default=None, help='Judge model')
    parser.add_argument('--judge-args', type=str, default=None, help='Judge arguments in JSON')
    parser.add_argument('--api-nproc', type=int, default=4, help='Parallel API calls')
    parser.add_argument('--retry', type=int, default=3, help='Retry count')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    # Setup logger
    logger = get_logger('EVAL_ONLY')
    
    model_name = args.model
    pred_root = osp.join(args.work_dir, model_name)
    
    for dataset_name in args.data:
        logger.info(f'Evaluating {model_name} x {dataset_name}')
        
        # Step 1: Build dataset
        try:
            dataset = build_dataset(dataset_name)
        except Exception as e:
            logger.error(f'Failed to build dataset {dataset_name}: {e}')
            continue
        
        # Step 2: Find prediction file
        result_file_base = f'{model_name}_{dataset_name}.xlsx'
        result_file = osp.join(pred_root, result_file_base)
        if not osp.exists(result_file):
            logger.error(f'Prediction file not found: {result_file}')
            continue
        
        logger.info(f'Loading predictions from: {result_file}')
        
        # Step 3: Build judge kwargs
        judge_kwargs = {
            'nproc': args.api_nproc,
            'verbose': args.verbose,
            'retry': args.retry,
            **(json.loads(args.judge_args) if args.judge_args else {}),
        }
        
        if args.judge is not None:
            judge_kwargs['model'] = args.judge
        else:
            judge_kwargs['model'] = 'gpt-5.4'
        
        logger.info(f'Judge kwargs: {judge_kwargs}')
        
        # Step 4: Perform evaluation ONLY
        try:
            logger.info(f'Starting evaluation...')
            eval_results = dataset.evaluate(result_file, **judge_kwargs)
            
            if eval_results is not None:
                logger.info(f'Evaluation results:')
                if isinstance(eval_results, dict):
                    logger.info(json.dumps(eval_results, indent=4))
                else:
                    logger.info(eval_results)
            else:
                logger.info(f'No evaluation results returned')
        except Exception as e:
            logger.error(f'Evaluation failed: {e}')
            import traceback, sys
            traceback.print_exc()
            sys.exit(1)

if __name__ == '__main__':
    main()

