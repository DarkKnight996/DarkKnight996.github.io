#!/usr/bin/env bash

set -euo pipefail

usage() {
  cat <<'EOF'
Generate a weekly daily-progress markdown template under docs/daily_progress.

Usage:
  bash generate_daily_progress_template.sh [YYYY-MM-DD]

Arguments:
  YYYY-MM-DD   Optional reference date. The script generates the Monday-Sunday
               template for the week containing this date.

Examples:
  bash generate_daily_progress_template.sh
  bash generate_daily_progress_template.sh 2026-04-19
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

REFERENCE_DATE="${1:-$(date +%Y-%m-%d)}"

if ! date -j -f "%Y-%m-%d" "$REFERENCE_DATE" "+%Y-%m-%d" >/dev/null 2>&1; then
  echo "Invalid date: $REFERENCE_DATE" >&2
  echo "Expected format: YYYY-MM-DD" >&2
  exit 1
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR="$REPO_ROOT/docs/daily_progress"
mkdir -p "$OUTPUT_DIR"

WEEKDAY="$(date -j -f "%Y-%m-%d" "$REFERENCE_DATE" "+%u")"
MONDAY_OFFSET=$((WEEKDAY - 1))

MONDAY_DATE="$(date -j -v-"${MONDAY_OFFSET}"d -f "%Y-%m-%d" "$REFERENCE_DATE" "+%Y-%m-%d")"
SUNDAY_DATE="$(date -j -v+6d -f "%Y-%m-%d" "$MONDAY_DATE" "+%Y-%m-%d")"

START_COMPACT="$(date -j -f "%Y-%m-%d" "$MONDAY_DATE" "+%Y%m%d")"
END_COMPACT="$(date -j -f "%Y-%m-%d" "$SUNDAY_DATE" "+%Y%m%d")"
OUTPUT_FILE="$OUTPUT_DIR/${START_COMPACT}-${END_COMPACT}.md"

if [[ -f "$OUTPUT_FILE" ]]; then
  echo "File already exists: $OUTPUT_FILE" >&2
  exit 1
fi

SECTIONS=()
for INDEX in 1 2 3 4 5 6 7; do
  DAY_OFFSET=$((INDEX - 1))
  DAY_DATE="$(date -j -v+"${DAY_OFFSET}"d -f "%Y-%m-%d" "$MONDAY_DATE" "+%Y-%m-%d")"
  SECTIONS+=("## ${INDEX}. ${DAY_DATE}")
done

{
  echo "# ${START_COMPACT}-${END_COMPACT}"
  echo
  for SECTION in "${SECTIONS[@]}"; do
    echo "$SECTION"
    echo
    echo
  done
  echo "## 总结"
  echo
} > "$OUTPUT_FILE"

echo "Created: $OUTPUT_FILE"
