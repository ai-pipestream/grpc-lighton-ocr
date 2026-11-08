"""Ensure the acceptance gate rejects false hardware and output claims."""
import copy
import unittest

from hardware_rpc_test import validate_output, validate_placement


def placement():
    return {"stages": [dict(stage="INFERENCE_STAGE_" + name,
                            backend="COMPUTE_BACKEND_CUDA", deviceName="test device",
                            deviceId="CUDA0", runtimeVerified=True, cpuCompute=False)
                       for name in ("VISION", "DECODER")]}


def output():
    provenance = placement()
    events = []
    text = "INVOICE 1042 125.00 Paid"
    for number in (1, 2):
        base = {"text": text, "selfRef": f"#/texts/{number - 1}", "prov": [{
            "pageNo": number, "charspan": {"end": len(text)},
            "bbox": {"l": 10, "r": 500, "t": 10, "b": 220}}]}
        events.append({"pageDocument": {"pageNo": number, "provenance": provenance,
            "finishReason": "FINISH_REASON_END_TOKEN", "document": {
                "texts": [{"text": {"base": base}}],
                "generationTraces": [{"steps": [{"token": 1}]}]}}})
    texts = [event["pageDocument"]["document"]["texts"][0] for event in events]
    events += [{"document": {"texts": texts, "body": {"children": [
        {"ref": f"#/texts/{n}"} for n in (0, 1)]}}},
        {"complete": {"pages": 2, "pagesOk": 2}}]
    return events


class HardwareChecks(unittest.TestCase):
    def test_hardware_claims(self):
        validate_placement(placement(), "cuda")
        for field, value in (("runtimeVerified", False), ("cpuCompute", True),
                             ("backend", "COMPUTE_BACKEND_CPU"), ("deviceName", "")):
            with self.subTest(field=field):
                broken = placement()
                broken["stages"][0][field] = value
                with self.assertRaises(ValueError):
                    validate_placement(broken, "cuda")
        unknown = placement()
        del unknown["stages"][0]["cpuCompute"]
        with self.assertRaises(ValueError):
            validate_placement(unknown, "cuda")

    def test_missing_vision(self):
        broken = placement()
        broken["stages"] = [broken["stages"][1]]
        with self.assertRaises(ValueError):
            validate_placement(broken, "cuda")

    def test_real_content_and_geometry_required(self):
        validate_output(output(), placement())
        for failure in ("text", "box", "shifted_box", "whole_page_box", "partial", "page", "trailer", "reference"):
            with self.subTest(failure=failure):
                events = copy.deepcopy(output())
                page = events[1]["pageDocument"]
                base = page["document"]["texts"][0]["text"]["base"]
                if failure == "text":
                    base["text"] = "nonempty but wrong"
                elif failure == "box":
                    base["prov"][0]["bbox"]["r"] = 10
                elif failure == "shifted_box":
                    base["prov"][0]["bbox"] = {"l": 600, "r": 950, "t": 10, "b": 220}
                elif failure == "whole_page_box":
                    base["prov"][0]["bbox"] = {"l": 0, "r": 1000, "t": 0, "b": 300}
                elif failure == "partial":
                    page["finishReason"] = "FINISH_REASON_TOKEN_LIMIT"
                elif failure == "page":
                    base["prov"][0]["pageNo"] = 1
                elif failure == "trailer":
                    events[-1]["complete"]["pagesOk"] = 1
                else:
                    events[-2]["document"]["body"]["children"].reverse()
                with self.assertRaises(ValueError):
                    validate_output(events, placement())


if __name__ == "__main__":
    unittest.main()
