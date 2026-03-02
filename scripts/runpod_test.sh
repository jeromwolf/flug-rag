#!/bin/bash
# =============================================================================
# RunPod RAG Quality Test - 10 OCR Questions
# =============================================================================
# Tests RAG quality with curated questions across different document types.
# No Python dependencies needed - pure bash + curl.
#
# Usage:
#   bash scripts/runpod_test.sh [BASE_URL]
#   bash scripts/runpod_test.sh http://localhost:8000
# =============================================================================

set -e

BASE_URL="${1:-http://localhost:8000}"
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

PASS=0
WARN=0
FAIL=0
TOTAL=0

echo ""
echo "============================================"
echo "  flux-rag RAG Quality Test (10 Questions)"
echo "  Server: $BASE_URL"
echo "============================================"
echo ""

# Health check
echo -n "Health check... "
HEALTH=$(curl -s -o /dev/null -w "%{http_code}" "$BASE_URL/health" 2>/dev/null)
if [ "$HEALTH" = "200" ]; then
    echo -e "${GREEN}OK${NC}"
else
    echo -e "${RED}FAIL (HTTP $HEALTH)${NC}"
    echo "Server not responding. Start it first: bash /workspace/flux-rag/start.sh"
    exit 1
fi
echo ""

# Test function
# Args: question_num, question, expected_keywords (comma-separated), negative_keywords
run_test() {
    local NUM="$1"
    local QUESTION="$2"
    local EXPECTED="$3"
    local NEGATIVE="$4"
    local CATEGORY="$5"

    TOTAL=$((TOTAL + 1))
    echo -e "${CYAN}[Q$NUM]${NC} $QUESTION"
    echo -e "      Category: $CATEGORY"

    # Send SSE request and collect response
    RESPONSE=""
    SOURCES=""

    # Use curl to send SSE request and parse chunks
    RAW=$(curl -s -N --max-time 120 \
        -H "Content-Type: application/json" \
        -d "{\"message\": \"$QUESTION\", \"session_id\": \"test-q$NUM-$(date +%s)\"}" \
        "$BASE_URL/api/chat/stream" 2>/dev/null)

    # Extract answer chunks
    RESPONSE=$(echo "$RAW" | grep "^data:" | grep '"content"' | \
        sed 's/^data: //g' | \
        python3 -c "
import sys, json
text = ''
for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    try:
        d = json.loads(line)
        if 'content' in d:
            text += d['content']
    except:
        pass
print(text)
" 2>/dev/null || echo "")

    # Extract source filenames
    SOURCES=$(echo "$RAW" | grep "^data:" | grep '"filename"' | \
        sed 's/^data: //g' | \
        python3 -c "
import sys, json
files = []
for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    try:
        d = json.loads(line)
        if 'filename' in d:
            score = d.get('score', 0)
            files.append(f\"{d['filename']} ({score:.3f})\")
    except:
        pass
for f in files[:3]:
    print(f'      Source: {f}')
" 2>/dev/null || echo "")

    # Truncate response for display
    DISPLAY_RESP=$(echo "$RESPONSE" | head -c 200)
    echo "      Answer: ${DISPLAY_RESP}..."

    if [ -n "$SOURCES" ]; then
        echo "$SOURCES"
    fi

    # Check expected keywords
    FOUND=0
    FOUND_LIST=""
    MISSING_LIST=""
    IFS=',' read -ra KEYWORDS <<< "$EXPECTED"
    TOTAL_KW=${#KEYWORDS[@]}

    for kw in "${KEYWORDS[@]}"; do
        kw_trimmed=$(echo "$kw" | xargs)
        if echo "$RESPONSE" | grep -q "$kw_trimmed"; then
            FOUND=$((FOUND + 1))
            FOUND_LIST="$FOUND_LIST [$kw_trimmed]"
        else
            MISSING_LIST="$MISSING_LIST [$kw_trimmed]"
        fi
    done

    # Check negative keywords (should NOT appear)
    NEG_FOUND=""
    if [ -n "$NEGATIVE" ]; then
        IFS=',' read -ra NEG_KEYWORDS <<< "$NEGATIVE"
        for nkw in "${NEG_KEYWORDS[@]}"; do
            nkw_trimmed=$(echo "$nkw" | xargs)
            if echo "$RESPONSE" | grep -q "$nkw_trimmed"; then
                NEG_FOUND="$NEG_FOUND [$nkw_trimmed]"
            fi
        done
    fi

    # Empty response check
    if [ -z "$RESPONSE" ] || [ ${#RESPONSE} -lt 10 ]; then
        echo -e "      Result: ${RED}FAIL${NC} (empty or too short response)"
        FAIL=$((FAIL + 1))
        echo ""
        return
    fi

    # Refusal check
    if echo "$RESPONSE" | grep -qE "확인되지 않|찾을 수 없|제공된.*없|답변.*어렵"; then
        echo -e "      Result: ${RED}FAIL${NC} (LLM refused to answer)"
        FAIL=$((FAIL + 1))
        echo ""
        return
    fi

    # Negative keyword check
    if [ -n "$NEG_FOUND" ]; then
        echo -e "      Result: ${YELLOW}WARN${NC} (negative keywords found:$NEG_FOUND)"
        WARN=$((WARN + 1))
        echo ""
        return
    fi

    # Score based on keyword match ratio
    if [ "$TOTAL_KW" -gt 0 ]; then
        RATIO=$((FOUND * 100 / TOTAL_KW))
    else
        RATIO=100
    fi

    if [ "$RATIO" -ge 80 ]; then
        echo -e "      Result: ${GREEN}PASS${NC} ($FOUND/$TOTAL_KW keywords matched)"
        PASS=$((PASS + 1))
    elif [ "$RATIO" -ge 50 ]; then
        echo -e "      Result: ${YELLOW}WARN${NC} ($FOUND/$TOTAL_KW keywords, missing:$MISSING_LIST)"
        WARN=$((WARN + 1))
    else
        echo -e "      Result: ${RED}FAIL${NC} ($FOUND/$TOTAL_KW keywords, missing:$MISSING_LIST)"
        FAIL=$((FAIL + 1))
    fi

    echo ""
}

# =============================================================================
# Test Questions
# =============================================================================

# Q1: 한국가스공사법 제1조 (법률 기본)
run_test 1 \
    "한국가스공사법 제1조(목적)의 내용은 무엇인가요?" \
    "가스,장기적,안정,공급,국민생활,편익,공공복리" \
    "" \
    "법률-기본"

# Q2: 한국가스기술공사 기업개요 (Entity 구분)
run_test 2 \
    "한국가스기술공사는 어떤 회사인가요?" \
    "가스기술공사,안전,검사" \
    "한국가스공사법" \
    "기업정보"

# Q3: 정관 내용 (정관 문서)
run_test 3 \
    "한국가스기술공사 정관에서 회사의 목적사업은 무엇인가요?" \
    "목적,사업,가스" \
    "" \
    "정관"

# Q4: 내부규정 겸직 (규정 내용)
run_test 4 \
    "내부규정에서 겸직에 관한 규정은 어떻게 되어 있나요?" \
    "겸직,허가,승인" \
    "" \
    "내부규정"

# Q5: 감사규정 (내부규정 상세)
run_test 5 \
    "감사규정에서 감사의 종류는 무엇이 있나요?" \
    "종합감사,특별감사,일상감사" \
    "" \
    "내부규정"

# Q6: 자본금 (ALIO 숫자 팩트)
run_test 6 \
    "한국가스공사의 자본금은 얼마인가요?" \
    "자본" \
    "" \
    "ALIO-팩트"

# Q7: 도시가스사업법 제1조 (교차 법률)
run_test 7 \
    "도시가스사업법 제1조(목적)의 내용은 무엇인가요?" \
    "도시가스사업,합리적,조정,육성,공급,원활,공공의 안전" \
    "" \
    "법률-교차"

# Q8: 고압가스안전관리법 (교차 법률)
run_test 8 \
    "고압가스 안전관리법의 목적은 무엇인가요?" \
    "고압가스,위해,방지,공공의 안전" \
    "" \
    "법률-교차"

# Q9: 도시가스 정의 (정의 직접 인용)
run_test 9 \
    "도시가스사업법에서 '도시가스'의 정의는 무엇인가요?" \
    "천연가스,액화,연료용 가스" \
    "" \
    "법률-정의"

# Q10: 고압가스 정의 (정의 직접 인용)
run_test 10 \
    "고압가스 안전관리법에서 규정하는 고압가스의 기준은 무엇인가요?" \
    "고압가스,압력,온도" \
    "" \
    "법률-정의"

# =============================================================================
# Summary
# =============================================================================
echo "============================================"
echo "  Test Results Summary"
echo "============================================"
echo ""
echo -e "  ${GREEN}PASS${NC}: $PASS"
echo -e "  ${YELLOW}WARN${NC}: $WARN"
echo -e "  ${RED}FAIL${NC}: $FAIL"
echo -e "  TOTAL: $TOTAL"
echo ""

SCORE=$((PASS * 100 / TOTAL))
if [ "$SCORE" -ge 80 ]; then
    echo -e "  Score: ${GREEN}${SCORE}%${NC} - Ready for demo!"
elif [ "$SCORE" -ge 60 ]; then
    echo -e "  Score: ${YELLOW}${SCORE}%${NC} - Needs improvement"
else
    echo -e "  Score: ${RED}${SCORE}%${NC} - Critical issues"
fi

echo ""
echo "  Model: $(curl -s http://localhost:11434/api/tags 2>/dev/null | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    models = [m['name'] for m in data.get('models', [])]
    print(', '.join(models))
except:
    print('unknown')
" 2>/dev/null || echo "unknown")"
echo ""
echo "============================================"
