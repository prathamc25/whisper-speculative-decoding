import argparse
from benchmark import run_speculative_benchmark, run_beam_benchmark
from api import run_server
from utils import check_and_install_dependencies

if __name__ == "__main__":
    check_and_install_dependencies()
    
    parser = argparse.ArgumentParser(description="Whisper Speculative Decoding Suite")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Benchmark Command
    bench_parser = subparsers.add_parser("benchmark", help="Run performance benchmarks")
    bench_parser.add_argument("--mode", choices=["speculative", "beam"], default="speculative", help="Type of benchmark to run")
    bench_parser.add_argument("--samples", type=int, default=8, help="Number of audio samples to test")
    
    # API Command
    api_parser = subparsers.add_parser("api", help="Start the REST API server")
    api_parser.add_argument("--port", type=int, default=8000, help="Port to run the server on")
    
    args = parser.parse_args()
    
    if args.command == "benchmark":
        if args.mode == "speculative":
            run_speculative_benchmark(num_samples=args.samples)
        elif args.mode == "beam":
            run_beam_benchmark(num_samples=args.samples)
    elif args.command == "api":
        run_server(port=args.port)
    else:
        parser.print_help()