import argparse
import runpy
import sys
from pathlib import Path


def run_script(script_path: Path, argv=None):
    original_argv = sys.argv[:]
    try:
        sys.argv = [str(script_path)] + (argv or [])
        runpy.run_path(str(script_path), run_name='__main__')
    finally:
        sys.argv = original_argv


def main():
    parser = argparse.ArgumentParser(description='Run training and/or batch inference in one command')
    parser.add_argument('--train', choices=['none', 'resnet', 'densenet', 'both'], default='none')
    parser.add_argument('--infer-model', choices=['resnet', 'densenet', 'both'], default='both')
    parser.add_argument('--input', default='', help='Image file/folder for batch inference')
    parser.add_argument('--output', default='', help='Optional CSV path for batch inference output')
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent

    if args.train in {'resnet', 'both'}:
        print('\n=== Training: ResNet ===')
        run_script(repo_root / 'src' / 'train' / 'run_resnet_train.py')

    if args.train in {'densenet', 'both'}:
        print('\n=== Training: DenseNet ===')
        run_script(repo_root / 'src' / 'train' / 'run_densenet_train.py')

    if args.input:
        print('\n=== Batch Inference ===')
        infer_argv = ['--model', args.infer_model, '--input', args.input]
        if args.output:
            infer_argv.extend(['--output', args.output])
        run_script(repo_root / 'src' / 'inference' / 'run_batch_inference.py', infer_argv)
    else:
        print('\nNo inference input provided. Skipping batch inference.')


if __name__ == '__main__':
    main()
