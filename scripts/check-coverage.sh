#!/usr/bin/env bash
set -euo pipefail

swift test --enable-code-coverage

BIN_PATH="$(swift build --show-bin-path)"
TEST_BUNDLE="${BIN_PATH}/RLSwiftPackageTests.xctest"
PROFDATA="${BIN_PATH}/codecov/default.profdata"

if [[ -d "${TEST_BUNDLE}" ]]; then
  TEST_BINARY="${TEST_BUNDLE}/Contents/MacOS/RLSwiftPackageTests"
else
  TEST_BINARY="${TEST_BUNDLE}"
fi

REPORT="$(xcrun llvm-cov report "${TEST_BINARY}" \
  -instr-profile "${PROFDATA}" \
  -ignore-filename-regex='(^\.build|/\.build|/Tests/|Package.swift)')"

printf '%s\n' "${REPORT}"

TOTAL_LINE="$(printf '%s\n' "${REPORT}" | awk '/TOTAL/ { line=$0 } END { print line }')"
REGION_COVERAGE="$(printf '%s\n' "${TOTAL_LINE}" | awk '{ print $4 }' | tr -d '%')"

if [[ "${REGION_COVERAGE}" != "100.00" ]]; then
  printf 'Coverage gate failed: expected 100.00%% region coverage, got %s%%\n' "${REGION_COVERAGE}" >&2
  exit 1
fi
