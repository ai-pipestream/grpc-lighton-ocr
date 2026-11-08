#!/usr/bin/env python3
"""Cancel a valid model stream after its first page, then assert real recovery."""
import argparse
import base64
import json
import os
import pathlib
import select
import signal
import subprocess
import time

import hardware_rpc_test as checks


def payload(pages):
    image = (checks.ROOT / "tests/fixtures/native-invoice.png").read_bytes()
    return "\n".join([json.dumps({"options": {
        "model": "LIGHTON_MODEL_0_8B", "pageCount": pages, "temperature": 0,
        "emitPageDocuments": True, "emitDocument": True}})] + [
        json.dumps({"page": {"pageNo": n, "png": base64.b64encode(image).decode()}})
        for n in range(1, pages + 1)])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--address", required=True)
    parser.add_argument("--backend", choices=("cpu", "cuda", "sycl", "metal"), required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    checks.require(args.repeats >= 3, "at least three cancellation repetitions required")
    args.output.mkdir(parents=True, exist_ok=False)
    for repeat in range(args.repeats):
        command = ["grpcurl", "-plaintext", "-max-time", "300", "-import-path",
                   str(checks.ROOT / "proto"), "-proto",
                   "ai/pipestream/lightonocr/v1/lighton_ocr.proto", "-d", "@",
                   args.address, checks.SERVICE + "ParseDocument"]
        raw = b""
        with (args.output / f"cancel-{repeat}.stderr").open("wb") as errors:
            process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                       stderr=errors)
            try:
                process.stdin.write(payload(8).encode())
                process.stdin.close()
                deadline = time.monotonic() + 180
                first_page = None
                while time.monotonic() < deadline and process.poll() is None:
                    readable, _, _ = select.select([process.stdout], [], [], 1)
                    if not readable:
                        continue
                    chunk = os.read(process.stdout.fileno(), 65536)
                    if not chunk:
                        break
                    raw += chunk
                    rest = raw.decode()
                    while rest.strip():
                        try:
                            event, end = json.JSONDecoder().raw_decode(rest.lstrip())
                        except json.JSONDecodeError:
                            break
                        rest = rest.lstrip()[end:]
                        if "pageDocument" in event:
                            first_page = event
                            break
                    if first_page:
                        break
                checks.require(first_page is not None, "valid first page must complete before cancellation")
                # With seven uploaded pages pending, allow the next inference to begin.
                time.sleep(0.2)
                checks.require(process.poll() is None, "stream must still be active")
                process.send_signal(signal.SIGINT)
                exit_code = process.wait(timeout=10)
                (args.output / f"cancel-{repeat}.exitcode").write_text(str(exit_code) + "\n")
                checks.require(exit_code != 0, "cancelled stream must exit nonzero")
                raw += process.stdout.read()
                checks.require(b'"complete"' not in raw, "cancel before document completion")
            finally:
                if process.poll() is None:
                    process.kill()
                    process.wait()
                (args.output / f"cancel-{repeat}.jsonl").write_bytes(raw)
        # A real model response is required: retry only admission while cancellation drains.
        deadline = time.monotonic() + 15
        while True:
            try:
                recovered, seconds = checks.rpc(args.address, "ParseDocument", payload(2), 180)
                break
            except subprocess.CalledProcessError as error:
                if "ResourceExhausted" not in error.stderr or time.monotonic() >= deadline:
                    raise
                time.sleep(0.1)
        (args.output / f"recovery-{repeat}.jsonl").write_text(recovered)
        info, _ = checks.rpc(args.address, "GetServiceInfo", "{}", 30)
        provenance = json.loads(info)["provenance"]
        checks.validate_placement(provenance, args.backend)
        checks.validate_output(checks.decode_events(recovered), provenance)
        print(f"PASS valid-stream cancellation/recovery {repeat}: {seconds:.3f}s", flush=True)


if __name__ == "__main__":
    main()
