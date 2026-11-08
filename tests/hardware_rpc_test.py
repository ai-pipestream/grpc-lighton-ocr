#!/usr/bin/env python3
"""Compare real accelerator and CPU RPC output on the same host.

Both servers must already be running. This gate does not launch a model,
select a backend, or treat requested configuration as execution evidence.
JSON is grpcurl's diagnostic representation of the protobuf wire, never an
inference transport. Output artifacts belong to the test client only.
"""
import argparse
import base64
import hashlib
import json
import pathlib
import statistics
import subprocess
import time

ROOT = pathlib.Path(__file__).resolve().parent.parent
SERVICE = "ai.pipestream.lightonocr.v1.LightOnOcrService/"
BACKENDS = {name: "COMPUTE_BACKEND_" + name.upper()
            for name in ("cpu", "cuda", "sycl", "metal")}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def decode_events(text):
    result = []
    decoder = json.JSONDecoder()
    while text.strip():
        text = text.lstrip()
        event, end = decoder.raw_decode(text)
        result.append(event)
        text = text[end:]
    return result


def validate_placement(provenance, backend):
    stages = provenance.get("stages", [])
    expected = {"INFERENCE_STAGE_VISION", "INFERENCE_STAGE_DECODER"}
    require(len(stages) == 2 and {s.get("stage") for s in stages} == expected,
            "vision and decoder must each report placement exactly once")
    for stage in stages:
        require(stage.get("backend") == BACKENDS[backend],
                f"unexpected stage backend: {stage}")
        require(stage.get("runtimeVerified") is True,
                "configuration alone does not prove runtime placement")
        require(bool(stage.get("deviceName")) and bool(stage.get("deviceId")),
                "native device identity is required")
        require(isinstance(stage.get("cpuCompute"), bool),
                "CPU model computation must be explicitly disclosed")
        require(stage["cpuCompute"] == (backend == "cpu"),
                "this gate requires both model stages on the selected device")


def validate_output(events, provenance):
    require(bool(events) and "complete" in events[-1], "missing final trailer")
    require(not any("pageFailure" in event for event in events), "page failure")
    trailer = events[-1]["complete"]
    require(trailer.get("pages") == 2 and trailer.get("pagesOk") == 2
            and trailer.get("pagesFailed", 0) == 0, "wrong page counts")
    pages = [event["pageDocument"] for event in events if "pageDocument" in event]
    require(len(pages) == 2, "missing page documents")
    for number, page in enumerate(pages, 1):
        require(page.get("pageNo") == number, "page ordering changed")
        require(page.get("provenance") == provenance, "page runtime identity changed")
        require(page.get("finishReason") == "FINISH_REASON_END_TOKEN",
                "partial output is not successful inference")
        document = page["document"]
        bases = [next(iter(item.values()))["base"] for item in document.get("texts", [])]
        text = " ".join(base["text"] for base in bases)
        require(all(word in text for word in ("INVOICE", "1042", "125.00", "Paid")),
                f"invoice content missing on page {number}: {text}")
        for base in bases:
            require(bool(base.get("prov")), "text has no location")
            for prov in base["prov"]:
                box = prov["bbox"]
                require(prov.get("pageNo") == number, "wrong box page")
                require(0 <= box.get("l", 0) < box["r"] <= 1000
                        and 0 <= box.get("t", 0) < box["b"] <= 300,
                        "empty or out-of-page box")
                require(prov["charspan"].get("start", 0) == 0
                        and prov["charspan"]["end"] == len(base["text"]),
                        "invalid Unicode span")
        traces = document.get("generationTraces", [])
        require(bool(traces) and bool(traces[0].get("steps")), "missing raw generation")
    folded = [event["document"] for event in events if "document" in event]
    require(len(folded) == 1, "expected one folded document")
    refs = [next(iter(item.values()))["base"]["selfRef"]
            for item in folded[0].get("texts", [])]
    require(bool(refs) and len(refs) == len(set(refs)), "duplicate folded references")
    require([child["ref"] for child in folded[0]["body"]["children"]] == refs,
            "folded reading order differs from page order")


def rpc(address, method, payload, timeout):
    command = ["grpcurl", "-plaintext", "-max-time", str(timeout),
               "-import-path", str(ROOT / "proto"), "-proto",
               "ai/pipestream/lightonocr/v1/lighton_ocr.proto", "-d", "@",
               address, SERVICE + method]
    start = time.perf_counter()
    result = subprocess.run(command, input=payload, text=True, capture_output=True,
                            timeout=timeout + 5, check=True)
    return result.stdout, time.perf_counter() - start


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", required=True, help="accelerator server host:port")
    parser.add_argument("--cpu", required=True, help="explicit CPU server on the same host")
    parser.add_argument("--backend", choices=("cuda", "sycl", "metal"), required=True)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--timeout", type=float, default=300)
    parser.add_argument("--minimum-speedup", type=float, default=1.0)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    options = parser.parse_args()
    require(options.repeats >= 3, "at least three measured repetitions required")
    require(options.minimum_speedup >= 1, "accelerator must not be slower than CPU")
    options.output.mkdir(parents=True, exist_ok=False)
    image = (ROOT / "tests/fixtures/native-invoice.png").read_bytes()
    payload = "\n".join([json.dumps({"options": {
        "model": "LIGHTON_MODEL_0_8B", "pageCount": 2, "temperature": 0,
        "emitPageDocuments": True, "emitDocument": True}})] + [
            json.dumps({"page": {"pageNo": number,
                                 "png": base64.b64encode(image).decode()}})
            for number in (1, 2)])
    infos = {}
    samples = {"cpu": [], "candidate": []}
    for label, address, backend in (("cpu", options.cpu, "cpu"),
                                    ("candidate", options.candidate, options.backend)):
        raw, _ = rpc(address, "GetServiceInfo", "{}", options.timeout)
        info = json.loads(raw)
        require(info.get("engineLoaded") is True, f"{label} model not loaded")
        infos[label] = info
        (options.output / f"{label}-info.json").write_text(raw)
    for key in ("modelSha256", "projectorSha256", "modelRevision", "backendRevision",
                "serviceVersion", "serverInstance"):
        require(infos["cpu"]["provenance"].get(key)
                and infos["cpu"]["provenance"].get(key)
                == infos["candidate"]["provenance"].get(key),
                f"CPU and accelerator must share {key}")
    # Alternate runs to reduce temporal bias; first pair is warm-up only.
    for iteration in range(options.repeats + 1):
        order = ("cpu", "candidate") if iteration % 2 == 0 else ("candidate", "cpu")
        for label in order:
            raw, seconds = rpc(getattr(options, label), "ParseDocument", payload, options.timeout)
            (options.output / f"{label}-{iteration}.jsonl").write_text(raw)
            if iteration == 0:
                # A cold process cannot claim observed execution until it
                # has actually evaluated the model. Verify after warm-up.
                info_raw, _ = rpc(getattr(options, label), "GetServiceInfo", "{}", options.timeout)
                observed = json.loads(info_raw)
                for key in ("modelSha256", "projectorSha256", "modelRevision",
                            "backendRevision", "serviceVersion", "serverInstance"):
                    require(observed["provenance"].get(key) == infos[label]["provenance"].get(key),
                            f"{label} identity changed during warm-up: {key}")
                validate_placement(observed["provenance"],
                                   "cpu" if label == "cpu" else options.backend)
                infos[label] = observed
                (options.output / f"{label}-observed-info.json").write_text(info_raw)
            validate_output(decode_events(raw), infos[label]["provenance"])
            if iteration:
                samples[label].append(seconds)
            print(f"{label} run={iteration} seconds={seconds:.3f} content=PASS", flush=True)
    speedup = statistics.median(samples["cpu"]) / statistics.median(samples["candidate"])
    receipt = {"fixture_sha256": hashlib.sha256(image).hexdigest(),
               "measured_two_page_rpc_seconds": samples, "median_speedup": speedup,
               "minimum_speedup": options.minimum_speedup, "info": infos,
               "scope": "one invoice fixture repeated; not a general OCR quality or memory soak test"}
    (options.output / "comparison.json").write_text(json.dumps(receipt, indent=2) + "\n")
    require(speedup >= options.minimum_speedup,
            f"speedup {speedup:.3f} below required {options.minimum_speedup:.3f}")
    print(f"PASS: verified stage placement, repeated content and geometry; speedup={speedup:.3f}")


if __name__ == "__main__":
    main()
