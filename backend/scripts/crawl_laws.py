"""
한국 법령 크롤링 스크립트

국가법령정보센터 Open API를 사용하여 가스 관련 법령을 수집합니다.

사용법:
1. 법제처에서 Open API 키 발급: https://open.law.go.kr
2. 환경변수 설정: export LAW_API_KEY=your_api_key
3. 실행: python scripts/crawl_laws.py
"""

import asyncio
import json
import os
import re
from pathlib import Path
from typing import Any

import httpx

# API 설정
LAW_API_BASE = "https://open.law.go.kr/openapi/lsInfoP.do"
API_KEY = os.getenv("LAW_API_KEY", "")

# 수집할 법령 목록
TARGET_LAWS = [
    "한국가스공사법",
    "한국가스공사법 시행령",
    "도시가스사업법",
    "도시가스사업법 시행령",
    "도시가스사업법 시행규칙",
    "고압가스 안전관리법",
    "고압가스 안전관리법 시행령",
    "액화석유가스의 안전관리 및 사업법",
]


async def search_law(client: httpx.AsyncClient, law_name: str) -> dict | None:
    """법령 검색 API 호출."""
    params = {
        "OC": API_KEY,
        "target": "law",
        "type": "XML",
        "query": law_name,
    }

    try:
        resp = await client.get(LAW_API_BASE.replace("lsInfoP", "lawSearch"), params=params)
        resp.raise_for_status()
        # XML 파싱 필요
        return {"raw": resp.text}
    except Exception as e:
        print(f"Error searching {law_name}: {e}")
        return None


async def get_law_content(client: httpx.AsyncClient, law_id: str) -> dict | None:
    """법령 본문 조회 API 호출."""
    params = {
        "OC": API_KEY,
        "target": "law",
        "type": "JSON",
        "LsId": law_id,
    }

    try:
        resp = await client.get(LAW_API_BASE, params=params)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        print(f"Error fetching law {law_id}: {e}")
        return None


def clean_text(text: str) -> str:
    """법령 텍스트 정리."""
    # HTML 태그 제거
    text = re.sub(r'<[^>]+>', '', text)
    # 공백 정리
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


async def main():
    """메인 크롤링 함수."""
    if not API_KEY:
        print("=" * 60)
        print("국가법령정보센터 Open API 사용 안내")
        print("=" * 60)
        print()
        print("1. API 키 발급:")
        print("   https://open.law.go.kr 에서 회원가입 후 API 키 발급")
        print()
        print("2. 환경변수 설정:")
        print("   export LAW_API_KEY=your_api_key")
        print()
        print("3. 스크립트 실행:")
        print("   python scripts/crawl_laws.py")
        print()
        print("=" * 60)
        print()
        print("API 키 없이도 수동 데이터를 사용할 수 있습니다.")
        print("data/sample_dataset/한국가스공사법/ 폴더의 파일을 확인하세요.")
        return

    output_dir = Path(__file__).parent.parent / "data/sample_dataset/법령"
    output_dir.mkdir(parents=True, exist_ok=True)

    async with httpx.AsyncClient(timeout=30) as client:
        for law_name in TARGET_LAWS:
            print(f"Searching: {law_name}")
            result = await search_law(client, law_name)
            if result:
                # 결과 저장
                filename = law_name.replace(" ", "_") + ".json"
                with open(output_dir / filename, "w", encoding="utf-8") as f:
                    json.dump(result, f, ensure_ascii=False, indent=2)
                print(f"  Saved: {filename}")

            await asyncio.sleep(1)  # API 호출 제한


if __name__ == "__main__":
    asyncio.run(main())
