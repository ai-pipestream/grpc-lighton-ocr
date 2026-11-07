#!/usr/bin/env bash
# Boot-proofs a grpc-lighton-ocr image. The native engine is not linked, so
# the image boots with no weights: the server must come up, bind its
# listener, log its "listening on" line, and shut down cleanly on SIGTERM.
set -euo pipefail

usage() {
  echo "Usage: $0 IMAGE" >&2
  exit 64
}
[[ $# -eq 1 ]] || usage
image=$1

echo "== smoke: library closure of the shipped binary"
trace=$(docker run --rm -e LD_TRACE_LOADED_OBJECTS=1 \
    --entrypoint /usr/local/bin/grpc-lighton-ocr-server "$image" 2>&1 || true)
if ! grep -q '=>' <<<"$trace"; then
  echo "the loader printed no dependency list for the server in $image:" >&2
  echo "$trace" >&2
  exit 1
fi
if grep -q "not found" <<<"$trace"; then
  echo "unresolved shared libraries for the server in $image:" >&2
  grep "not found" <<<"$trace" >&2
  exit 1
fi

echo "== smoke: image runs as the non-root user 65532"
image_user=$(docker inspect --format '{{.Config.User}}' "$image")
if [[ "$image_user" != "65532:65532" ]]; then
  echo "expected USER 65532:65532, image has '${image_user:-root}'" >&2
  exit 1
fi

echo "== smoke: image exposes only the gRPC port"
exposed=$(docker inspect --format '{{range $port, $_ := .Config.ExposedPorts}}{{$port}} {{end}}' "$image")
if [[ "$exposed" != "50074/tcp " ]]; then
  echo "expected EXPOSE 50074 only, image exposes '${exposed}'" >&2
  exit 1
fi

echo "== smoke: server boots and reaches its listener"
container="lighton-ocr-smoke-$$"
docker run -d --read-only --name "$container" "$image" >/dev/null
trap 'docker rm -f "$container" >/dev/null 2>&1 || true' EXIT
for _ in $(seq 1 60); do
  docker logs "$container" 2>&1 | grep -q "listening on" && break
  sleep 1
done
boot_output=$(docker logs "$container" 2>&1 || true)
echo "$boot_output"
if grep -q "error while loading shared libraries" <<<"$boot_output"; then
  echo "the loader failed before main ran" >&2
  exit 1
fi
if ! grep -q "grpc-lighton-ocr listening on" <<<"$boot_output"; then
  echo "server did not reach its listener; logs above" >&2
  exit 1
fi

echo "== smoke: SIGTERM shuts the server down cleanly"
docker stop "$container" >/dev/null
exit_code=$(docker wait "$container" || true)
if [[ "$exit_code" != "0" ]]; then
  echo "expected exit code 0 after SIGTERM, got '${exit_code:-unknown}'" >&2
  exit 1
fi
trap - EXIT
docker rm -f "$container" >/dev/null 2>&1 || true

echo "smoke-test: OK ($image)"
