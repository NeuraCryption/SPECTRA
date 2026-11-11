import sys
import socket
import argparse
import threading
import webbrowser
import time
from pathlib import Path
from typing import Optional

from spectra.hardware import HardwareInspector
from spectra.compressor import SpectraCompressor
from spectra.orchestrator import Spectra

def find_available_port(default_port: int = 7860, max_tries: int = 20) -> int:
    for p in range(default_port, default_port + max_tries):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(('127.0.0.1', p)) != 0:
                return p
    return default_port

def launch_ui_server(host: str = "127.0.0.1", port: int = 7860, model: Optional[str] = None, no_browser: bool = False):
    import uvicorn
    from spectra.server import app, discover_models, api_engine_load, LoadModelRequest

    chosen_port = find_available_port(port)

    # 1. Hardware probe
    profile = HardwareInspector.probe()

    # 2. Discover local models
    models_data = discover_models()
    comp_count = len(models_data.get("compressed_models", []))
    raw_count = len(models_data.get("raw_models", []))

    print("\n" + "=" * 70)
    print("  SPECTRA: Universal Model Compression & GPU Serving Engine")
    print("=" * 70)
    print(f"  Target Device : {profile.device_name} ({profile.total_vram_mb:.0f} MB VRAM)")
    print(f"  Web Dashboard : http://{host}:{chosen_port}")
    print(f"  Local Models  : {comp_count} ready-to-run compressed, {raw_count} raw checkpoints")
    print("=" * 70 + "\n")

    # If model specified, auto-load
    if model:
        m_path = Path(model).resolve()
        if m_path.is_dir() and (m_path / "model.trdc").exists():
            m_path = m_path / "model.trdc"
        print(f"[SPECTRA] Pre-loading model: {m_path.name}...")
        try:
            api_engine_load(LoadModelRequest(trdc_path=str(m_path)))
            print(f"[SPECTRA] Model successfully mounted in GPU VRAM.")
        except Exception as e:
            print(f"[SPECTRA] Notice: Could not pre-load model: {e}")

    # Launch browser thread
    if not no_browser:
        def _open():
            time.sleep(1.2)
            try:
                webbrowser.open(f"http://{host}:{chosen_port}")
            except Exception:
                pass
        threading.Thread(target=_open, daemon=True).start()

    # Run FastAPI via Uvicorn
    uvicorn.run("spectra.server:app", host=host, port=chosen_port, log_level="info")

def main():
    parser = argparse.ArgumentParser(
        prog="spectra",
        description="SPECTRA: Model Compression & GPU Serving Engine"
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Command: ui (default)
    parser_ui = subparsers.add_parser("ui", aliases=["serve", "web"], help="Launch localhost Web UI dashboard")
    parser_ui.add_argument("--host", type=str, default="127.0.0.1", help="Host interface to bind (default: 127.0.0.1)")
    parser_ui.add_argument("--port", type=int, default=7860, help="Port to listen on (default: 7860)")
    parser_ui.add_argument("--model", type=str, default=None, help="Auto-load model container or folder on launch")
    parser_ui.add_argument("--no-browser", action="store_true", help="Do not open browser automatically")

    # Command: status
    parser_status = subparsers.add_parser("status", help="Probe and display hardware telemetry card")

    # Command: compress
    parser_comp = subparsers.add_parser("compress", help="Compress a HuggingFace model into .trdc container")
    parser_comp.add_argument("--model", type=str, required=True, help="Path or HuggingFace ID of model")
    parser_comp.add_argument("--output", type=str, default=None, help="Output .trdc container path")
    parser_comp.add_argument("--group-size", type=int, default=32, help="Microscaling group size (default: 32)")
    parser_comp.add_argument("--outliers", type=float, default=0.01, help="Base outlier ratio (default: 0.01 = 1%%)")

    # Command: run
    parser_run = subparsers.add_parser("run", help="Serve a model and generate text via GPU")
    parser_run.add_argument("--model", type=str, required=True, help="Path to .trdc container or model directory")
    parser_run.add_argument("--prompt", type=str, default="Solve 3x + 15 = 42 step-by-step.", help="Input prompt")
    parser_run.add_argument("--max-tokens", type=int, default=1024, help="Max tokens to generate (default: 1024)")
    parser_run.add_argument("--batch-test", action="store_true", help="Also run batched throughput benchmark (B=2, B=4)")

    # If no arguments provided, or if flags passed without a subcommand (e.g. spectra --port 7860)
    if len(sys.argv) == 1:
        launch_ui_server()
        return
    elif len(sys.argv) > 1 and sys.argv[1].startswith("-") and sys.argv[1] not in ["-h", "--help"]:
        sys.argv.insert(1, "ui")

    args = parser.parse_args()

    if args.command in ["ui", "serve", "web"]:
        launch_ui_server(
            host=args.host,
            port=args.port,
            model=args.model,
            no_browser=args.no_browser
        )

    elif args.command == "status":
        profile = HardwareInspector.probe()
        print(profile.summary_card())

    elif args.command == "compress":
        Spectra.compress(
            model_id_or_path=args.model,
            output_container=args.output,
            group_size=args.group_size,
            outlier_ratio=args.outliers
        )

    elif args.command == "run":
        engine = Spectra.serve(args.model)

        # Format with chat template if available
        prompt = args.prompt
        tok = engine.tokenizer
        if tok is not None and hasattr(tok, 'apply_chat_template') and getattr(tok, 'chat_template', None):
            try:
                messages = [{"role": "user", "content": prompt}]
                prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            except Exception:
                pass

        print(f"\n[Prompt]: {args.prompt}")
        res = engine.generate(prompt, max_new_tokens=args.max_tokens)
        print("\n" + "=" * 80)
        print("SPECTRA GENERATED RESPONSE:")
        print("=" * 80)
        print(res.text)
        print("=" * 80)
        print(f"Summary: {res.summary()}")

        if args.batch_test:
            print("\n[Running Batched Throughput Benchmark (B=2, B=4)]...")
            prompts_b2 = [args.prompt, "Explain the core concept of quantum superposition in one sentence."]
            b2 = engine.generate_batch(prompts_b2, max_new_tokens=32)
            print(f"B=2 Aggregate Throughput: {b2['aggregate_tok_per_sec']:.2f} tok/s | Latency: {b2['step_latency_ms']:.2f} ms")

            prompts_b4 = prompts_b2 + ["Solve for x: x^2 = 64.", "Describe why GPU register memory is faster than global VRAM."]
            b4 = engine.generate_batch(prompts_b4, max_new_tokens=32)
            print(f"B=4 Aggregate Throughput: {b4['aggregate_tok_per_sec']:.2f} tok/s | Latency: {b4['step_latency_ms']:.2f} ms")

    else:
        parser.print_help()

if __name__ == "__main__":
    main()
