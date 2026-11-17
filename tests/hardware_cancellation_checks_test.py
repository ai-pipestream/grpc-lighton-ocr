#!/usr/bin/env python3
"""Keep observation-off regression separate from strict device qualification."""
import copy
import unittest
from hardware_cancellation_test import argument_parser, validate_metadata


def placement(verified):
    return {"stages": [{"stage": "INFERENCE_STAGE_" + stage,
                        "backend": "COMPUTE_BACKEND_METAL", "deviceName": "Apple M2",
                        "deviceId": "MTL0", "runtimeVerified": verified, "cpuCompute": False}
                       for stage in ("VISION", "DECODER")]}


class ModeChecks(unittest.TestCase):
    def test_strict_default(self):
        args = argument_parser().parse_args(["--address", "localhost:1", "--backend", "metal", "--output", "out"])
        self.assertFalse(args.unobserved_regression)
        validate_metadata(placement(True), "metal")
        with self.assertRaises(ValueError):
            validate_metadata(placement(False), "metal")

    def test_off_requires_unverified(self):
        validate_metadata(placement(False), "metal", True)
        omitted = placement(False)
        for stage in omitted["stages"]:
            del stage["runtimeVerified"]
        validate_metadata(omitted, "metal", True)
        for index in (0, 1):
            invalid = placement(False)
            invalid["stages"][index]["runtimeVerified"] = True
            with self.assertRaises(ValueError):
                validate_metadata(invalid, "metal", True)

    def test_off_requires_stage_identities(self):
        for field in ("stage", "backend", "deviceName", "deviceId"):
            invalid = copy.deepcopy(placement(False))
            invalid["stages"][0][field] = ""
            with self.assertRaises(ValueError):
                validate_metadata(invalid, "metal", True)


if __name__ == "__main__":
    unittest.main()
