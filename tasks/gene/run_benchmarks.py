#!/usr/bin/env python
"""
CTM Gene Benchmark Runner

This script provides a unified interface to run all CTM gene benchmarks.

Benchmarks:
1. Perturb-seq: Tests knockout prediction against experimental data
2. BEELINE: Evaluates GRN inference against ChIP-seq validated edges
3. Trajectory: GSEA analysis of CTM's internal "thinking" trajectory
4. PertEval: Standardized perturbation prediction evaluation
5. CausalBench: Causal GRN inference with real Perturb-seq interventional data

Usage:
    # Run all benchmarks
    python run_benchmarks.py --all --checkpoint path/to/model.pt

    # Run specific benchmark
    python run_benchmarks.py --benchmark beeline --dataset hESC
    python run_benchmarks.py --benchmark perturbseq --checkpoint model.pt
    python run_benchmarks.py --benchmark trajectory --checkpoint model.pt
    python run_benchmarks.py --benchmark causalbench --dataset weissmann_k562
"""
import argparse
import os
import sys
import subprocess

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))


def parse_args():
    parser = argparse.ArgumentParser(description='CTM Gene Benchmark Runner')
    parser.add_argument('--benchmark', type=str, default='all',
                        choices=['all', 'perturbseq', 'beeline', 'trajectory', 'perteval', 'causalbench'],
                        help='Which benchmark to run')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to trained CTM checkpoint')
    parser.add_argument('--dataset', type=str, default='hESC',
                        help='Dataset for BEELINE/CausalBench benchmark')
    parser.add_argument('--output_dir', type=str, default='logs/benchmarks',
                        help='Base output directory')
    parser.add_argument('--device', type=str, default='cpu',
                        help='Device to use')
    parser.add_argument('--training_iterations', type=int, default=400,
                        help='Training iterations for benchmarks')
    parser.add_argument('--download_data', action='store_true',
                        help='Download benchmark datasets')
    return parser.parse_args()


def download_data():
    """Download all benchmark datasets."""
    print("="*60)
    print("DOWNLOADING BENCHMARK DATASETS")
    print("="*60)
    
    # Hematopoiesis
    print("\n--- Hematopoiesis (CELLxGENE) ---")
    try:
        from data.hematopoiesis_data import download_hematopoiesis_data, process_hematopoiesis_trajectories
        if not os.path.exists('data/hematopoiesis_processed.pt'):
            download_hematopoiesis_data()
            process_hematopoiesis_trajectories()
        else:
            print("Already downloaded.")
    except Exception as e:
        print(f"Error: {e}")
    
    # Norman Perturb-seq
    print("\n--- Norman et al. 2019 Perturb-seq ---")
    try:
        from data.perturbseq_data import download_norman_data, process_norman_data
        if not os.path.exists('data/norman_processed.pt'):
            download_norman_data()
            process_norman_data()
        else:
            print("Already downloaded.")
    except Exception as e:
        print(f"Error: {e}")
    
    # BEELINE
    print("\n--- BEELINE (hESC, mESC) ---")
    try:
        from data.beeline_data import download_beeline_data, process_beeline_for_ctm
        for ds in ['hESC', 'mESC']:
            if not os.path.exists(f'data/beeline/{ds}_processed.pt'):
                download_beeline_data(ds)
                process_beeline_for_ctm(ds)
            else:
                print(f"{ds} already downloaded.")
    except Exception as e:
        print(f"Error: {e}")
    
    print("\n" + "="*60)
    print("DATA DOWNLOAD COMPLETE")
    print("="*60)


def run_beeline(args):
    """Run BEELINE GRN benchmark."""
    print("\n" + "="*60)
    print("BEELINE GRN BENCHMARK")
    print("="*60)
    
    from tasks.gene.benchmark_beeline import run_benchmark
    
    class BeelineArgs:
        dataset = args.dataset
        output_dir = os.path.join(args.output_dir, f'beeline_{args.dataset}')
        device = args.device
        training_iterations = 2000
        lr = 1e-3
        batch_size = 32
    
    results = run_benchmark(BeelineArgs())
    return results


def run_perturbseq(args):
    """Run Perturb-seq benchmark."""
    print("\n" + "="*60)
    print("PERTURB-SEQ BENCHMARK")
    print("="*60)
    
    if args.checkpoint is None:
        print("Error: --checkpoint required for perturbseq benchmark")
        return None
    
    cmd = [
        'python', 'tasks/gene/benchmark_perturbseq.py',
        '--checkpoint', args.checkpoint,
        '--output_dir', os.path.join(args.output_dir, 'perturbseq'),
        '--device', args.device,
    ]
    
    subprocess.run(cmd)


def run_trajectory(args):
    """Run trajectory GSEA benchmark."""
    print("\n" + "="*60)
    print("TRAJECTORY GSEA BENCHMARK")
    print("="*60)
    
    cmd = [
        'python', 'tasks/gene/benchmark_trajectory.py',
        '--output_dir', os.path.join(args.output_dir, 'trajectory'),
        '--device', args.device,
        '--dataset', 'meninges',
    ]
    
    if args.checkpoint:
        cmd.extend(['--checkpoint', args.checkpoint])
    
    subprocess.run(cmd)


def run_perteval(args):
    """Run PertEval benchmark."""
    print("\n" + "="*60)
    print("PERTEVAL-SCFM BENCHMARK")
    print("="*60)
    
    if args.checkpoint is None:
        print("Error: --checkpoint required for perteval benchmark")
        return None
    
    # Check for perturb data
    perturb_data = 'data/norman_processed.pt'
    if not os.path.exists(perturb_data):
        print(f"Error: Perturbation data not found at {perturb_data}")
        print("Run with --download_data first.")
        return None
    
    cmd = [
        'python', 'tasks/gene/benchmark_perteval.py',
        '--checkpoint', args.checkpoint,
        '--perturb_data', perturb_data,
        '--output_dir', os.path.join(args.output_dir, 'perteval'),
        '--device', args.device,
    ]
    
    subprocess.run(cmd)


def run_causalbench(args):
    """Run CausalBench benchmark."""
    print("\n" + "="*60)
    print("CAUSALBENCH BENCHMARK")
    print("="*60)
    
    from tasks.gene.benchmark_causalbench import run_benchmark_manual
    
    class CausalBenchArgs:
        dataset = args.dataset if args.dataset in ['weissmann_k562', 'weissmann_rpe1'] else 'weissmann_k562'
        training_regime = 'observational'
        output_dir = os.path.join(args.output_dir, f'causalbench_{dataset}')
        data_dir = 'data/causalbench'
        device = args.device
        training_iterations = args.training_iterations
        lr = 1e-3
        batch_size = 32
        n_genes = 200
        subset_data = 1.0
    
    results = run_benchmark_manual(CausalBenchArgs())
    return results


def main():
    args = parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.download_data:
        download_data()
        return
    
    if args.benchmark == 'all':
        # Run all benchmarks
        print("\n" + "#"*60)
        print("# RUNNING ALL CTM GENE BENCHMARKS")
        print("#"*60)
        
        # BEELINE first (trains its own model)
        run_beeline(args)
        
        # CausalBench (also trains its own model)
        run_causalbench(args)
        
        # Others need a checkpoint
        if args.checkpoint:
            run_perturbseq(args)
            run_trajectory(args)
            run_perteval(args)
        else:
            print("\nSkipping perturbseq, trajectory, perteval (no checkpoint provided)")
        
        print("\n" + "#"*60)
        print("# ALL BENCHMARKS COMPLETE")
        print("#"*60)
        
    elif args.benchmark == 'beeline':
        run_beeline(args)
    elif args.benchmark == 'perturbseq':
        run_perturbseq(args)
    elif args.benchmark == 'trajectory':
        run_trajectory(args)
    elif args.benchmark == 'perteval':
        run_perteval(args)
    elif args.benchmark == 'causalbench':
        run_causalbench(args)


if __name__ == '__main__':
    main()

