"""Run the supplied 10,000-particle example from any working directory."""
from pathlib import Path
import argparse

if __package__ in (None, ''):
    from main import run_particles
else:
    from .main import run_particles


def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=root/'outputs')
    parser.add_argument('--reference', choices=['centroid', 'particle'], default='centroid')
    parser.add_argument('--no-preview', action='store_true')
    args = parser.parse_args()
    result = run_particles(root/'example'/'particles.txt', root/'example'/'domain.json',
                           args.output_dir, args.reference, preview=not args.no_preview)
    summary = result['summary']
    print(f"Cells: {summary['count']}")
    print(f"Ed: {summary['Ed']:.8f}")
    print(f"Ad: {summary['orientation_all_cells']['Ad']:.8f}")
    print(f"Results: {args.output_dir.resolve()}")


if __name__ == '__main__':
    main()
