#!/usr/bin/env python3
"""Real Metal RPC cancellation/recovery; uses an already-running loopback server."""
import argparse
import base64
import json
import pathlib
import subprocess
import time

from hardware_rpc_test import ROOT, SERVICE, decode_events, rpc, validate_output, validate_placement


def payload(image, pages):
    return "\n".join([json.dumps({"options": {"model": "LIGHTON_MODEL_0_8B",
        "pageCount": pages, "temperature": 0, "emitPageDocuments": True,
        "emitDocument": True}})] + [json.dumps({"page": {"pageNo": n,
        "png": base64.b64encode(image).decode()}}) for n in range(1, pages + 1)])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--address", required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(exist_ok=False, parents=True)
    image = (ROOT / "tests/fixtures/native-invoice.png").read_bytes()
    warm, _ = rpc(args.address, "ParseDocument", payload(image, 2), 120)
    info, _ = rpc(args.address, "GetServiceInfo", "{}", 10)
    provenance = json.loads(info)["provenance"]
    validate_placement(provenance, "metal")
    validate_output(decode_events(warm), provenance)
    (args.output / "warm.jsonl").write_text(warm)
    (args.output / "info.json").write_text(info)
    for repeat in range(3):
        command = ["grpcurl", "-plaintext", "-max-time", "0.25", "-import-path", str(ROOT / "proto"),
            "-proto", "ai/pipestream/lightonocr/v1/lighton_ocr.proto", "-d", "@",
            args.address, SERVICE + "ParseDocument"]
        start = time.monotonic()
        cancelled = subprocess.run(command, input=payload(image, 64), text=True,
            capture_output=True, timeout=10)
        elapsed = time.monotonic() - start
        (args.output / f"cancel-{repeat}.stdout").write_text(cancelled.stdout)
        (args.output / f"cancel-{repeat}.stderr").write_text(cancelled.stderr)
        assert cancelled.returncode != 0 and "deadline" in cancelled.stderr.lower(), cancelled.stderr
        assert elapsed < 10, "cancellation did not return promptly"
        # A prior kernel can finish cooperatively; bound the recovery wait.
        end = time.monotonic() + 60
        attempt = 0
        while True:
            recovery, seconds = rpc(args.address, "ParseDocument", payload(image, 2), 120)
            (args.output / f"recovery-{repeat}-{attempt}.jsonl").write_text(recovery)
            events = decode_events(recovery)
            failures = [event["pageFailure"] for event in events if "pageFailure" in event]
            if not failures:
                validate_output(events, provenance)
                break
            assert time.monotonic() < end, "inference admission did not recover after cancellation"
            assert all("busy" in str(failure).lower() for failure in failures), failures
            attempt += 1
            time.sleep(0.1)
        print(f"cycle={repeat} cancellation_seconds={elapsed:.3f} recovery_rpc_seconds={seconds:.3f} PASS", flush=True)


if __name__ == "__main__":
    main()
