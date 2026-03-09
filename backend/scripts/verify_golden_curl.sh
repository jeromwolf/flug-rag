#!/bin/bash
# Verify golden dataset against RunPod RAG API using curl
# Usage: bash scripts/verify_golden_curl.sh

API_BASE="https://7rzubyo9fsfmco-8000.proxy.runpod.net"

# Login
echo "=== Login ==="
TOKEN=$(curl -s -X POST "$API_BASE/api/auth/login" \
  -H "Content-Type: application/json" \
  -d '{"username":"admin","password":"admin123"}' | python3 -c "import sys,json; print(json.load(sys.stdin).get('access_token',''))" 2>/dev/null)

if [ -z "$TOKEN" ]; then
  echo "Login failed!"
  exit 1
fi
echo "Token acquired."

# Read questions from golden dataset
DATASET="tests/golden_dataset_evaluation.json"
TOTAL=$(python3 -c "import json; d=json.load(open('$DATASET')); print(len(d['questions']))")
echo "Total questions: $TOTAL"
echo ""

OK=0
MISS=0
ERR=0
TOTAL_TIME=0

printf "%-4s %-10s %-6s %-7s %s\n" "#" "Category" "Status" "Time" "Question"
printf "%-4s %-10s %-6s %-7s %s\n" "----" "----------" "------" "-------" "----------------------------------------"

for i in $(seq 0 $((TOTAL-1))); do
  # Extract question info
  QINFO=$(python3 -c "
import json
d = json.load(open('$DATASET'))
q = d['questions'][$i]
print(q['question'])
print(q['category'])
print(q['id'])
")
  QUESTION=$(echo "$QINFO" | head -1)
  CATEGORY=$(echo "$QINFO" | sed -n '2p')
  QID=$(echo "$QINFO" | sed -n '3p')

  # Escape question for JSON
  ESCAPED=$(python3 -c "import json; print(json.dumps('$QUESTION'))" 2>/dev/null || echo "\"$QUESTION\"")
  # Build JSON payload properly
  PAYLOAD=$(python3 -c "import json; print(json.dumps({'message': '$QUESTION', 'temperature': 0.1}))" 2>/dev/null)

  START=$(python3 -c "import time; print(time.time())")

  RESPONSE=$(curl -s --max-time 120 -X POST "$API_BASE/api/chat" \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer $TOKEN" \
    -d "$PAYLOAD" 2>/dev/null)

  END=$(python3 -c "import time; print(time.time())")
  ELAPSED=$(python3 -c "print(f'{$END - $START:.1f}')")

  # Check response
  if [ -n "$RESPONSE" ]; then
    SOURCES=$(echo "$RESPONSE" | python3 -c "import sys,json; d=json.load(sys.stdin); print(len(d.get('sources',[])))" 2>/dev/null || echo "0")
    ANSWER_LEN=$(echo "$RESPONSE" | python3 -c "import sys,json; d=json.load(sys.stdin); print(len(d.get('answer','')))" 2>/dev/null || echo "0")

    if [ "$ANSWER_LEN" -gt 10 ] 2>/dev/null; then
      STATUS="OK"
      OK=$((OK+1))
    elif [ "$CATEGORY" = "negative" ] && [ "$ANSWER_LEN" -gt 0 ] 2>/dev/null; then
      STATUS="OK"
      OK=$((OK+1))
    else
      STATUS="MISS"
      MISS=$((MISS+1))
    fi
  else
    STATUS="ERR"
    ERR=$((ERR+1))
  fi

  NUM=$((i+1))
  SHORT_Q=$(echo "$QUESTION" | cut -c1-35)
  printf "%-4s %-10s %-6s %5ss %s\n" "$NUM" "$CATEGORY" "$STATUS" "$ELAPSED" "$SHORT_Q"

  # Small delay
  sleep 0.3
done

echo ""
echo "============================================================"
echo "Results:"
echo "  OK:    $OK / $TOTAL ($(python3 -c "print(f'{$OK/$TOTAL*100:.1f}%')"))"
echo "  MISS:  $MISS / $TOTAL"
echo "  ERROR: $ERR / $TOTAL"
echo "============================================================"
