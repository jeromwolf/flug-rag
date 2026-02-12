# Terminology Service Integration

## Overview
Implemented a gas technology terminology dictionary service that automatically expands user queries with synonyms and related terms for improved search recall.

## Files Created/Modified

### 1. New File: `backend/rag/terminology.py`
Main terminology service module with:
- **TermEntry**: Dataclass for terminology entries (term, synonyms, related, english, category, definition)
- **ExpansionResult**: Dataclass for query expansion results
- **TerminologyService**: Core service class
  - `expand_query(query)`: Expands queries with synonyms
  - `lookup(term)`: Look up a specific term
  - `get_all_terms()`: Get all entries
  - `get_categories()`: Get category list
  - `search_terms(keyword)`: Search by keyword

### 2. Modified: `backend/rag/query_corrector.py`
- Added `__future__` annotations import for Python 3.9 compatibility
- Integrated terminology dictionary loading in `__init__`
  - Automatically loads synonyms from terminology service
  - Rebuilds correction patterns with terminology synonyms
- Added `expand_with_terminology()` method

### 3. Modified: `backend/rag/chain.py`
- Added terminology expansion step after query correction
- Uses expanded query for retrieval (search_query)
- Keeps original query for LLM prompt
- Updated both Multi-Query and standard retrieval paths to use search_query
- Added terminology_info to response metadata

### 4. Modified: `backend/rag/prompt.py`
- Added `__future__` annotations import for Python 3.9 compatibility

### 5. Created: `backend/data/glossary/gas_terminology.json`
Sample terminology dictionary with 15 gas industry terms:
- 정압기 (pressure regulator)
- 안전차단밸브 (emergency shut-off valve)
- 액화천연가스 (LNG)
- 액화석유가스 (LPG)
- 압축천연가스 (CNG)
- 가스배관망 (gas pipeline network)
- 한국가스공사 (KOGAS)
- 한국가스기술공사 (KGTC)
- 도시가스사업법
- 고압가스 안전관리법
- 액화석유가스의 안전관리 및 사업법
- 가스미터
- 정압소
- 가스누출경보기
- 메탄

## How It Works

### Query Flow
```
User Query: "정압기 설치기준은?"
    ↓
Query Correction (existing)
    ↓
Terminology Expansion (NEW)
    "정압기 설치기준은? 가스정압기 정압장치 레귤레이터 regulator"
    ↓
Retrieval (uses expanded query)
    ↓
LLM Generation (uses original query in prompt)
```

### Example Expansions
| Original Query | Expanded Query | Matched Terms |
|----------------|----------------|---------------|
| "정압기 설치기준" | "정압기 설치기준 가스정압기 정압장치 레귤레이터 regulator" | ["정압기"] |
| "LNG 저장탱크" | "LNG 저장탱크 액화천연가스 엘엔지" | ["액화천연가스"] |
| "가스공사 조직" | "가스공사 조직 한국가스공사 KOGAS 코가스" | ["한국가스공사"] |

### Response Metadata
When terminology expansion occurs, the response includes:
```json
{
  "metadata": {
    "terminology_expansion": {
      "matched_terms": ["정압기"],
      "expansions": [
        {
          "term": "정압기",
          "synonyms_added": ["가스정압기", "정압장치", "레귤레이터", "regulator"]
        }
      ]
    }
  }
}
```

## Configuration
The glossary location defaults to:
```
backend/data/glossary/gas_terminology.json
```

Can be overridden by passing `glossary_path` to `TerminologyService()`.

## Adding New Terms
Edit `backend/data/glossary/gas_terminology.json`:
```json
{
  "terms": [
    {
      "term": "용어명",
      "synonyms": ["동의어1", "동의어2"],
      "related": ["관련어1", "관련어2"],
      "english": "English Term",
      "category": "카테고리",
      "definition": "정의"
    }
  ]
}
```

## Benefits
1. **Improved Recall**: Finds documents using different terminology
2. **User-Friendly**: Users don't need to know exact technical terms
3. **Automatic**: No manual intervention required
4. **Transparent**: Metadata shows which terms were expanded
5. **Extensible**: Easy to add new terms to glossary

## Testing
```python
from rag.terminology import get_terminology_service

ts = get_terminology_service()
result = ts.expand_query("정압기 설치기준")
print(result.expanded_query)
# Output: "정압기 설치기준 가스정압기 정압장치 레귤레이터 regulator"
```

## Implementation Notes
1. Case-insensitive matching for English terms
2. Longest-first matching to handle multi-word terms
3. Avoids duplicate expansions
4. Original query preserved for LLM (to avoid confusing the model)
5. Expanded query used for retrieval (to improve recall)
6. Singleton pattern for efficient loading

## Future Enhancements
- Load glossary from database instead of JSON file
- Admin UI for managing terminology
- Term usage analytics
- Automatic term extraction from documents
- Multi-language support
- Context-aware expansion (different expansions for different domains)
