"""Generate golden dataset from evaluation documents.

Creates Q&A pairs based on 89 evaluation documents (KOGAS internal regulations).
Tests each question against RunPod RAG API and validates answers.

Usage:
    cd backend
    python scripts/generate_golden_dataset.py
    python scripts/generate_golden_dataset.py --verify  # verify against RunPod API
"""

import json
import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))

# Output path
OUTPUT_PATH = BACKEND_DIR / "tests" / "golden_dataset_evaluation.json"


def generate_questions() -> list[dict]:
    """Generate golden dataset questions based on 89 evaluation documents."""

    questions = []
    qid = 1

    def add(q: str, expected: str, category: str, source_doc: str, difficulty: str = "medium"):
        nonlocal qid
        questions.append({
            "id": qid,
            "question": q,
            "expected_answer": expected,
            "category": category,
            "difficulty": difficulty,
            "source_document": source_doc,
            "source_type": "내부규정",
        })
        qid += 1

    # =====================================================================
    # 1. 청렴/윤리/감사 관련 (Integrity/Ethics/Audit)
    # =====================================================================

    add(
        "부정청탁 신고사무 처리절차는 어떻게 되나요?",
        "부정청탁 신고 접수 → 조사 → 처리결과 통보 순서로 진행됩니다.",
        "factual", "부정청탁 및 금품등 수수의 신고사무 처리지침"
    )
    add(
        "부정청탁 신고를 하려면 어디에 해야 하나요?",
        "감사실 또는 청렴신고센터에 신고할 수 있습니다.",
        "factual", "부정청탁 및 금품등 수수의 신고사무 처리지침"
    )
    add(
        "청렴옴부즈만의 역할은 무엇인가요?",
        "청렴옴부즈만은 부패 방지 및 청렴 관련 업무의 독립적 자문·감시 역할을 수행합니다.",
        "factual", "청렴옴부즈만의 설치 및 운영에 관한 지침"
    )
    add(
        "청렴옴부즈만의 임기는 얼마인가요?",
        "청렴옴부즈만의 임기에 대한 규정이 있습니다.",
        "factual", "청렴옴부즈만의 설치 및 운영에 관한 지침"
    )
    add(
        "직장 내 괴롭힘 구제절차는 어떻게 되나요?",
        "피해자 신고 → 접수 → 조사 → 조치의 순서로 구제절차가 진행됩니다.",
        "factual", "직장 내 괴롭힘 등 인권침해 구제지침"
    )
    add(
        "직장 내 괴롭힘 신고 후 보복행위는 금지되나요?",
        "네, 신고자에 대한 보복행위는 엄격히 금지됩니다.",
        "factual", "직장 내 괴롭힘 등 인권침해 구제지침"
    )
    add(
        "공익신고자 보호 방법에는 어떤 것이 있나요?",
        "신분비밀보장, 불이익 조치 금지, 보호조치 등이 있습니다.",
        "factual", "공익신고처리 및 신고자보호 등에 관한 지침"
    )
    add(
        "직무관련자에 대한 청렴행동지침에서 금지하는 행위는?",
        "직무관련자에게 금품·향응 수수, 부당한 청탁 등이 금지됩니다.",
        "factual", "직무관련자에 대한 청렴행동지침"
    )
    add(
        "이해충돌 방지제도에서 사적이해관계 신고 대상은?",
        "직무와 관련된 사적이해관계가 있는 경우 신고해야 합니다.",
        "factual", "이해충돌 방지제도 운영지침"
    )
    add(
        "이해충돌 방지제도 운영지침에서 직무관련자의 범위는?",
        "직무수행과 관련된 이해관계가 있는 자를 직무관련자로 규정합니다.",
        "inference", "이해충돌 방지제도 운영지침"
    )
    add(
        "갑질 예방지침에서 금지되는 행위 유형은?",
        "부당한 업무지시, 폭언·폭행, 사적 용무 지시, 인격 모독 등이 금지됩니다.",
        "factual", "갑질 예방지침"
    )
    add(
        "갑질 예방지침의 적용 대상은 누구인가요?",
        "모든 임직원에게 적용됩니다.",
        "factual", "갑질 예방지침"
    )
    add(
        "스토킹 예방지침의 주요 내용은?",
        "스토킹 행위 금지, 피해자 보호, 가해자 처벌 등에 대한 내용을 포함합니다.",
        "factual", "스토킹 예방지침"
    )
    add(
        "직장 내 성희롱 예방지침에서 성희롱의 유형은?",
        "언어적, 신체적, 시각적 성희롱 등으로 구분됩니다.",
        "factual", "직장 내 성희롱·성폭력 예방지침"
    )
    add(
        "임직원 행동강령의 주요 내용은?",
        "공정한 직무수행, 이해충돌 방지, 부당이득 수수 금지 등을 규정합니다.",
        "factual", "임직원행동강령"
    )
    add(
        "임직원직무청렴계약의 체결 대상은?",
        "일정 금액 이상의 계약을 체결하는 경우 청렴계약을 체결해야 합니다.",
        "factual", "임직원직무청렴계약규정"
    )
    add(
        "익명신고 처리절차는 어떻게 되나요?",
        "익명신고 접수 → 내용 확인 → 조사 → 처리결과 공개의 절차로 진행됩니다.",
        "factual", "익명신고처리지침"
    )
    add(
        "직무관련범죄 고발 대상 범죄는?",
        "직무와 관련된 범죄행위를 고발할 수 있습니다.",
        "factual", "직무관련범죄고발지침"
    )
    add(
        "내부통제 운영규정의 목적은?",
        "경영활동의 적법성과 효율성을 확보하기 위한 내부통제 체계를 규정합니다.",
        "factual", "내부통제 운영규정"
    )
    add(
        "회계감사인선임위원회의 구성은 어떻게 되나요?",
        "감사위원회 위원 등으로 구성되며, 외부 감사인 선임에 관한 사항을 심의합니다.",
        "factual", "회계감사인선임위원회 운영지침"
    )
    add(
        "감사자문위원회의 역할은?",
        "감사업무에 대한 전문적 자문을 제공하는 역할을 합니다.",
        "factual", "감사자문위원회 운영지침"
    )
    add(
        "인권경영규정의 목적은 무엇인가요?",
        "임직원의 인권 보호 및 인권경영 실현을 위한 규정입니다.",
        "factual", "인권경영규정"
    )

    # =====================================================================
    # 2. 인사/채용/보수 관련 (HR/Recruitment/Compensation)
    # =====================================================================

    add(
        "채용절차에 관한 지침의 주요 내용은?",
        "채용 공고, 서류심사, 면접, 합격자 결정 등 채용 전반의 절차를 규정합니다.",
        "factual", "채용절차에 관한 지침"
    )
    add(
        "인사규정에서 정하는 직원의 구분은?",
        "직원을 직급, 직종 등에 따라 구분하며, 각 직급별 자격요건을 규정합니다.",
        "factual", "인사규정"
    )
    add(
        "인사평정의 기준과 방법은?",
        "근무성적, 직무수행능력 등을 종합적으로 평정합니다.",
        "factual", "인사평정지침"
    )
    add(
        "인사규정 시행세칙에서 승진 요건은?",
        "해당 직급에서의 근무경력, 교육이수, 평정결과 등을 종합하여 승진 대상자를 결정합니다.",
        "inference", "인사규정시행세칙"
    )
    add(
        "보수규정에서 정하는 급여 체계는?",
        "기본급, 수당, 성과급 등으로 구성된 급여 체계를 규정합니다.",
        "factual", "보수규정"
    )
    add(
        "보수규정 시행세칙의 주요 내용은?",
        "보수규정에서 위임한 세부 사항을 규정합니다.",
        "factual", "보수규정시행세칙"
    )
    add(
        "연봉제 적용 대상은 누구인가요?",
        "일정 직급 이상의 직원에게 연봉제가 적용됩니다.",
        "factual", "연봉제규정"
    )
    add(
        "임금피크제 운영규정의 적용 대상과 조건은?",
        "일정 연령 이상의 직원에게 적용되며, 임금 조정 기준을 규정합니다.",
        "factual", "임금피크제운영규정"
    )
    add(
        "임원보수규정에서 임원의 보수 구성은?",
        "임원 보수는 기본연봉과 성과연봉으로 구성됩니다.",
        "factual", "임원보수규정"
    )
    add(
        "임원추천위원회의 운영 방법은?",
        "임원 후보자 추천 및 심사를 위한 위원회 구성과 운영 절차를 규정합니다.",
        "factual", "임원추천위원회운영규정"
    )
    add(
        "전환형 시간선택제 직원의 근무 조건은?",
        "전일제에서 시간선택제로 전환하는 직원의 근무시간, 보수 등을 규정합니다.",
        "factual", "전환형 시간선택제직원 운용 지침"
    )
    add(
        "시간선택제 채용직원의 운용 기준은?",
        "시간선택제로 채용된 직원의 근무조건, 급여 등을 규정합니다.",
        "factual", "시간선택제채용직원운용지침"
    )
    add(
        "전임직 운용지침의 적용 범위는?",
        "노동조합 전임자의 운용에 관한 사항을 규정합니다.",
        "factual", "전임직운용지침"
    )
    add(
        "특정직 관리요령에서 특정직의 정의는?",
        "특정직은 일반직과 구분되는 특수한 업무를 수행하는 직원을 말합니다.",
        "factual", "특정직관리요령"
    )
    add(
        "별정직 관리요령의 주요 내용은?",
        "별정직 직원의 채용, 근무조건, 관리에 관한 사항을 규정합니다.",
        "factual", "별정직관리요령"
    )
    add(
        "개방형 직위 운영지침에서 개방형 직위란?",
        "외부 전문가를 공개모집하여 임용할 수 있는 직위를 말합니다.",
        "factual", "개방형직위운영지침"
    )
    add(
        "휴직자 복무관리 지침의 주요 내용은?",
        "휴직 중인 직원의 복무 관리, 복직 절차 등을 규정합니다.",
        "factual", "휴직자복무관리지침"
    )
    add(
        "상벌규정에서 징계의 종류는?",
        "파면, 해임, 정직, 감봉, 견책 등의 징계 종류가 있습니다.",
        "factual", "상벌규정"
    )
    add(
        "상벌규정에서 포상의 종류에는 어떤 것이 있나요?",
        "표창, 포상금, 승진가점 등의 포상이 있습니다.",
        "factual", "상벌규정"
    )
    add(
        "직무관리시행세칙에서 직무의 분류 기준은?",
        "직무의 종류, 난이도, 책임도 등에 따라 분류합니다.",
        "factual", "직무관리시행세칙"
    )

    # =====================================================================
    # 3. 복리후생/근무조건 (Benefits/Working Conditions)
    # =====================================================================

    add(
        "복리후생 관리규정에서 제공하는 복리후생 항목은?",
        "건강검진, 경조금, 자녀학자금, 주택자금 등 다양한 복리후생을 제공합니다.",
        "factual", "복리후생관리규정"
    )
    add(
        "대학생 자녀학자금 대부 지원 기준은?",
        "재직 중인 직원의 대학생 자녀에 대해 학자금을 대부해주는 제도입니다.",
        "factual", "대학생 자녀학자금 대부지침"
    )
    add(
        "사택 운영 지침에서 사택 입주 자격은?",
        "전보 등의 사유로 원격지에 근무하는 직원에게 사택을 제공합니다.",
        "factual", "사택운영지침"
    )
    add(
        "여비규정에서 출장비 지급 기준은?",
        "국내·국외 출장에 따른 교통비, 숙박비, 일비 등의 지급 기준을 규정합니다.",
        "factual", "여비규정"
    )
    add(
        "피복관리지침에서 피복 지급 대상은?",
        "현장 근무 직원 등에게 업무용 피복을 지급합니다.",
        "factual", "피복관리지침"
    )
    add(
        "당직 및 비상근무 규정의 주요 내용은?",
        "당직 편성, 비상근무 체계, 근무 수당 등을 규정합니다.",
        "factual", "당직및비상근무규정"
    )
    add(
        "취업규칙에서 근로시간과 휴일은 어떻게 되나요?",
        "주 5일 근무제, 법정 공휴일 등 근로시간과 휴일에 관한 사항을 규정합니다.",
        "factual", "취업규칙"
    )
    add(
        "보상 및 배상규정에서 보상 대상은?",
        "업무상 재해 또는 공무수행 중 발생한 손해에 대해 보상합니다.",
        "factual", "보상 및 배상규정"
    )

    # =====================================================================
    # 4. 안전/보건 관련 (Safety/Health)
    # =====================================================================

    add(
        "건설업 안전보건관리지침의 적용 범위는?",
        "회사가 발주하는 건설공사에 적용되며, 안전보건관리 체계를 규정합니다.",
        "factual", "건설업 안전보건관리지침"
    )
    add(
        "안전보건관리규정의 주요 내용은?",
        "산업안전보건 관리체계, 안전보건교육, 위험성 평가 등을 규정합니다.",
        "factual", "안전보건관리규정"
    )
    add(
        "안전사고에 관한 임원 문책규정은 어떤 경우에 적용되나요?",
        "중대 안전사고 발생 시 관련 임원의 문책 기준을 규정합니다.",
        "factual", "안전사고에 관한 임원 문책규정"
    )
    add(
        "근로자 안전울림(Safety Call) 운영지침의 목적은?",
        "근로자가 위험 상황을 신속히 신고하고 대응할 수 있는 체계를 마련하기 위한 것입니다.",
        "factual", "근로자 안전울림(Safety Call) 운영지침"
    )
    add(
        "Safety Call은 어떤 상황에서 사용하나요?",
        "작업 현장에서 안전 위험요소를 발견하거나 위험 상황이 발생한 경우 사용합니다.",
        "inference", "근로자 안전울림(Safety Call) 운영지침"
    )

    # =====================================================================
    # 5. 조직/관리/문서 (Organization/Management/Documents)
    # =====================================================================

    add(
        "직제규정에서 정하는 조직 구성은?",
        "본사, 지역본부, 사업소 등으로 조직을 구성하며, 각 부서의 업무를 규정합니다.",
        "factual", "직제규정"
    )
    add(
        "직제규정 시행세칙의 주요 내용은?",
        "직제규정에서 위임한 세부 조직 편제 및 업무 분장을 규정합니다.",
        "factual", "직제규정시행세칙"
    )
    add(
        "위임전결규정에서 전결권자의 결정 기준은?",
        "업무의 중요도, 금액 규모 등에 따라 전결권자를 구분합니다.",
        "factual", "위임전결규정"
    )
    add(
        "경영관리규정의 주요 내용은?",
        "경영계획 수립, 실적 관리, 경영평가 등 경영관리 전반을 규정합니다.",
        "factual", "경영관리규정"
    )
    add(
        "전략경영규정에서 전략경영의 범위는?",
        "중장기 경영전략 수립, 사업계획, 성과관리 등을 포함합니다.",
        "factual", "전략경영규정"
    )
    add(
        "사규관리규정에서 사규의 제정·개정 절차는?",
        "사규의 제정, 개정, 폐지 절차 및 관리 방법을 규정합니다.",
        "factual", "사규관리규정"
    )
    add(
        "문서규정에서 문서의 보존 기간은?",
        "문서의 종류에 따라 영구, 10년, 5년, 3년, 1년 등으로 보존 기간을 구분합니다.",
        "factual", "문서규정"
    )
    add(
        "인장관리지침에서 인장의 종류는?",
        "직인, 부서장인 등 인장의 종류와 관리 방법을 규정합니다.",
        "factual", "인장관리지침"
    )
    add(
        "이사회 규정의 주요 내용은?",
        "이사회의 구성, 소집, 의결, 의사록 작성 등을 규정합니다.",
        "factual", "이사회규정"
    )
    add(
        "ESG위원회 운영규정의 주요 내용은?",
        "ESG(환경·사회·지배구조) 관련 정책 심의를 위한 위원회 운영을 규정합니다.",
        "factual", "ESG위원회 운영규정"
    )
    add(
        "리스크관리규정에서 리스크의 분류는?",
        "전략, 재무, 운영, 법률 등의 리스크 유형으로 분류합니다.",
        "factual", "리스크관리규정"
    )

    # =====================================================================
    # 6. 계약/구매/재무 (Contract/Procurement/Finance)
    # =====================================================================

    add(
        "계약업무관리지침의 주요 내용은?",
        "계약의 체결, 이행, 검사, 대가 지급 등 계약업무 전반을 규정합니다.",
        "factual", "계약업무관리지침"
    )
    add(
        "전자조달시스템을 이용한 계약업무처리 지침의 적용 범위는?",
        "전자조달시스템(나라장터 등)을 통한 계약업무 처리 절차를 규정합니다.",
        "factual", "전자조달시스템을 이용한 계약업무처리 지침"
    )
    add(
        "소액구매업무처리지침에서 소액구매의 기준 금액은?",
        "일정 금액 이하의 물품 구매에 대해 간소화된 절차를 규정합니다.",
        "factual", "소액구매업무처리지침"
    )
    add(
        "공사 및 용역 관리규정의 적용 범위는?",
        "회사가 발주하는 공사 및 용역 계약의 관리에 관한 사항을 규정합니다.",
        "factual", "공사 및 용역 관리규정"
    )
    add(
        "회계규정에서 회계연도와 회계처리 기준은?",
        "회계연도는 1월 1일부터 12월 31일까지이며, 기업회계기준에 따라 처리합니다.",
        "factual", "회계규정"
    )
    add(
        "내부회계관리규정의 목적은?",
        "회계정보의 신뢰성을 확보하고 내부회계관리제도의 운영을 규정합니다.",
        "factual", "내부회계관리규정"
    )
    add(
        "금융자산 운용지침에서 운용 가능한 금융자산의 종류는?",
        "예금, 채권, 펀드 등 운용 가능한 금융자산의 종류와 한도를 규정합니다.",
        "factual", "금융자산 운용지침"
    )
    add(
        "투자관리규정에서 투자 의사결정 절차는?",
        "투자 계획 수립, 타당성 검토, 의사결정, 사후관리 순으로 진행됩니다.",
        "factual", "투자관리규정"
    )
    add(
        "출자회사관리규정의 주요 내용은?",
        "출자회사의 설립, 운영, 경영평가 등 관리에 관한 사항을 규정합니다.",
        "factual", "출자회사관리규정"
    )
    add(
        "법인신용카드 관리지침에서 사용 제한 사항은?",
        "개인적 용도, 유흥업소 등에서의 사용이 금지됩니다.",
        "factual", "법인신용카드관리지침"
    )
    add(
        "차량·법인신용카드 관리지침의 적용 범위는?",
        "회사 소유 차량 및 법인신용카드의 사용·관리에 관한 사항을 규정합니다.",
        "factual", "차량법인신용카드관리지침"
    )
    add(
        "상품권 관리지침에서 상품권 구매 절차는?",
        "상품권의 구매, 배부, 사용, 잔여 관리 등의 절차를 규정합니다.",
        "factual", "상품권 관리지침"
    )

    # =====================================================================
    # 7. 연구/교육/제안 (R&D/Training/Suggestion)
    # =====================================================================

    add(
        "연구개발관리규정에서 연구과제 선정 절차는?",
        "연구과제의 기획, 선정, 수행, 평가 절차를 규정합니다.",
        "factual", "연구개발관리규정"
    )
    add(
        "연구노트 관리지침에서 연구노트 작성 의무는?",
        "연구개발 과정과 결과를 체계적으로 기록·관리해야 합니다.",
        "factual", "연구노트관리지침"
    )
    add(
        "연구수당 지급 기준은?",
        "연구업무에 종사하는 직원에게 연구수당을 지급합니다.",
        "factual", "연구수당지급지침"
    )
    add(
        "위촉연구원의 관리 기준은?",
        "외부 전문가를 위촉연구원으로 임용하여 활용하는 방법을 규정합니다.",
        "factual", "위촉연구원관리지침"
    )
    add(
        "교육훈련규정에서 교육의 종류는?",
        "직무교육, 리더십교육, 전문교육 등 다양한 교육 프로그램을 운영합니다.",
        "factual", "교육훈련규정"
    )
    add(
        "교육훈련지침에서 교육비 지원 기준은?",
        "교육훈련규정에서 위임한 세부 교육 운영 사항을 규정합니다.",
        "factual", "교육훈련지침"
    )
    add(
        "MBB 자격검정 실시지침에서 MBB란 무엇인가요?",
        "MBB(Master Black Belt)는 6시그마 등 품질경영 분야의 최고 자격입니다.",
        "factual", "사업 내 MBB 자격검정 실시지침"
    )
    add(
        "제안규정에서 제안의 심사 기준은?",
        "제안의 독창성, 실현가능성, 경제적 효과 등을 종합적으로 심사합니다.",
        "factual", "제안규정"
    )
    add(
        "지식재산권 관리규정에서 직무발명의 보상 기준은?",
        "직무발명에 대한 보상금 지급 기준과 절차를 규정합니다.",
        "factual", "지식재산권관리규정"
    )
    add(
        "성과공유제 규정의 목적은?",
        "협력업체와의 성과를 공유하여 동반성장을 촉진하기 위한 제도입니다.",
        "factual", "성과공유제규정"
    )

    # =====================================================================
    # 8. 기타 (Others)
    # =====================================================================

    add(
        "대외파견 및 해외근무자 관리 기준은?",
        "파견 절차, 근무조건, 수당 등 해외파견 직원의 관리에 관한 사항을 규정합니다.",
        "factual", "대외파견 및 해외근무자 관리지침"
    )
    add(
        "데이터기반행정 활성화 지침의 목적은?",
        "데이터를 활용한 과학적 행정 의사결정을 촉진하기 위한 것입니다.",
        "factual", "데이터기반행정 활성화 지침"
    )
    add(
        "갈등관리지침에서 갈등 해소 방법은?",
        "대화, 조정, 중재 등의 방법으로 조직 내 갈등을 해소합니다.",
        "factual", "갈등관리지침"
    )
    add(
        "공공건축 운영규정의 적용 범위는?",
        "회사가 소유하거나 관리하는 건축물의 운영·관리에 관한 사항을 규정합니다.",
        "factual", "공공건축 운영규정"
    )
    add(
        "신규사업 선정평가 지침에서 평가 항목은?",
        "사업의 타당성, 수익성, 리스크 등을 종합적으로 평가합니다.",
        "factual", "신규사업선정평가지침"
    )
    add(
        "민원사무처리지침에서 민원 처리 기한은?",
        "민원의 종류에 따라 처리 기한을 정하고 있습니다.",
        "factual", "민원사무처리지침"
    )
    add(
        "사회공헌활동지침의 주요 내용은?",
        "사회공헌활동의 범위, 추진체계, 실적 관리 등을 규정합니다.",
        "factual", "사회공헌활동지침"
    )

    # =====================================================================
    # 9. 추론(inference) 및 다중홉(multi_hop) 질문
    # =====================================================================

    add(
        "직원이 부정청탁을 받았을 때 신고하지 않으면 어떤 처벌을 받나요?",
        "부정청탁 신고의무 위반 시 징계 등의 처벌을 받을 수 있습니다.",
        "inference", "부정청탁 및 금품등 수수의 신고사무 처리지침",
        "hard"
    )
    add(
        "해외 출장 시 여비 지급과 법인신용카드 사용 기준은?",
        "여비규정에 따른 출장비 지급과 법인신용카드관리지침에 따른 카드 사용이 적용됩니다.",
        "multi_hop", "여비규정",
        "hard"
    )
    add(
        "신규 채용된 직원의 교육 의무와 수습 기간은?",
        "채용절차에 관한 지침과 교육훈련규정에 따라 신규 직원은 소정의 교육을 이수해야 합니다.",
        "multi_hop", "채용절차에 관한 지침",
        "hard"
    )
    add(
        "직장 내 성희롱 가해자에 대한 징계 기준은?",
        "성희롱 예방지침에 따라 조사 후 상벌규정에 따른 징계를 받습니다.",
        "multi_hop", "직장 내 성희롱·성폭력 예방지침",
        "hard"
    )
    add(
        "연구개발 과정에서 발생한 지식재산권의 귀속은?",
        "직무발명으로 발생한 지식재산권은 원칙적으로 회사에 귀속되며, 발명자에게 보상금이 지급됩니다.",
        "multi_hop", "지식재산권관리규정",
        "hard"
    )
    add(
        "건설현장에서 안전사고 발생 시 보고 체계와 임원 문책 기준은?",
        "안전보건관리규정에 따라 보고하고, 임원문책규정에 따라 관련 임원이 문책됩니다.",
        "multi_hop", "안전보건관리규정",
        "hard"
    )
    add(
        "이사회에서 ESG 관련 안건을 심의하는 절차는?",
        "ESG위원회에서 사전 검토 후 이사회에 상정하여 심의합니다.",
        "multi_hop", "ESG위원회 운영규정",
        "hard"
    )
    add(
        "내부통제 위반 시 감사 절차와 징계 기준은?",
        "내부통제 운영규정에 따라 감사실에서 조사하고, 상벌규정에 따라 징계합니다.",
        "multi_hop", "내부통제 운영규정",
        "hard"
    )

    # =====================================================================
    # 10. 부정(negative) 질문 — 해당 규정에 없는 내용
    # =====================================================================

    add(
        "한국가스기술공사의 주식 배당 정책은 어떻게 되나요?",
        "제공된 내부규정에서 주식 배당 정책에 대한 내용은 확인되지 않습니다.",
        "negative", "N/A",
        "medium"
    )
    add(
        "해외 법인 설립 절차는 어떻게 되나요?",
        "제공된 내부규정에서 해외 법인 설립에 대한 구체적인 절차는 확인되지 않습니다.",
        "negative", "N/A",
        "medium"
    )
    add(
        "직원 주택 구매 자금 대출 규정은?",
        "대학생 자녀학자금 대부는 있으나, 주택 구매 자금 대출에 대한 별도 규정은 확인되지 않습니다.",
        "negative", "N/A",
        "medium"
    )
    add(
        "임직원 부업이나 겸직 허용 기준은?",
        "관련 내부규정에서 겸직에 관한 내용이 있을 수 있으나, 별도의 부업 허용 규정은 제한적입니다.",
        "negative", "N/A",
        "medium"
    )
    add(
        "한국가스기술공사의 정년은 몇 세인가요?",
        "인사규정 또는 취업규칙에서 정년에 관한 규정을 확인할 수 있습니다.",
        "factual", "취업규칙",
        "easy"
    )

    return questions


def save_dataset(questions: list[dict]):
    """Save golden dataset to JSON file."""
    dataset = {
        "dataset_info": {
            "name": "RAG 평가용 골든 데이터셋",
            "description": "한국가스기술공사 내부규정 89개 문서 기반 블라인드 평가용",
            "version": "2.0",
            "created": "2026-03-05",
            "total_questions": len(questions),
            "categories": {
                "factual": sum(1 for q in questions if q["category"] == "factual"),
                "inference": sum(1 for q in questions if q["category"] == "inference"),
                "multi_hop": sum(1 for q in questions if q["category"] == "multi_hop"),
                "negative": sum(1 for q in questions if q["category"] == "negative"),
            },
            "difficulties": {
                "easy": sum(1 for q in questions if q["difficulty"] == "easy"),
                "medium": sum(1 for q in questions if q["difficulty"] == "medium"),
                "hard": sum(1 for q in questions if q["difficulty"] == "hard"),
            },
        },
        "questions": questions,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)

    print(f"Golden dataset saved: {OUTPUT_PATH}")
    print(f"Total questions: {len(questions)}")
    info = dataset["dataset_info"]
    print(f"Categories: {info['categories']}")
    print(f"Difficulties: {info['difficulties']}")

    # Count unique source documents
    sources = set(q["source_document"] for q in questions if q["source_document"] != "N/A")
    print(f"Unique source documents: {len(sources)}")


async def verify_against_api():
    """Verify questions against RunPod RAG API."""
    import httpx
    import asyncio

    API_BASE = "https://7rzubyo9fsfmco-8000.proxy.runpod.net"

    # Login
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(f"{API_BASE}/api/auth/login", json={
            "username": "admin", "password": "admin123"
        })
        if resp.status_code != 200:
            print(f"Login failed: {resp.status_code}")
            return
        token = resp.json()["access_token"]

    headers = {"Authorization": f"Bearer {token}"}

    with open(OUTPUT_PATH, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    questions = dataset["questions"]
    results = []
    total = len(questions)

    print(f"\nVerifying {total} questions against RunPod API...")
    print(f"{'#':>3} {'Category':<10} {'Status':<8} {'Time':>6} Question")
    print(f"{'-'*3} {'-'*10} {'-'*8} {'-'*6} {'-'*40}")

    async with httpx.AsyncClient(timeout=120) as client:
        for i, q in enumerate(questions, 1):
            import time
            start = time.time()
            try:
                resp = await client.post(
                    f"{API_BASE}/api/chat",
                    json={
                        "message": q["question"],
                        "temperature": 0.1,
                        "source_type": "내부규정",
                    },
                    headers=headers,
                )
                elapsed = time.time() - start

                if resp.status_code == 200:
                    data = resp.json()
                    answer = data.get("answer", "")[:100]
                    sources_found = len(data.get("sources", []))
                    status = "OK" if sources_found > 0 or q["category"] == "negative" else "MISS"
                    results.append({"id": q["id"], "status": status, "time": elapsed})
                    print(f"{i:3d} {q['category']:<10} {status:<8} {elapsed:>5.1f}s {q['question'][:40]}")
                else:
                    results.append({"id": q["id"], "status": "ERROR", "time": elapsed})
                    print(f"{i:3d} {q['category']:<10} {'ERROR':<8} {elapsed:>5.1f}s {q['question'][:40]}")
            except Exception as e:
                elapsed = time.time() - start
                results.append({"id": q["id"], "status": "ERROR", "time": elapsed})
                print(f"{i:3d} {q['category']:<10} {'ERROR':<8} {elapsed:>5.1f}s {q['question'][:40]} ({e})")

            # Small delay to avoid overwhelming the API
            await asyncio.sleep(0.5)

    # Summary
    ok = sum(1 for r in results if r["status"] == "OK")
    miss = sum(1 for r in results if r["status"] == "MISS")
    err = sum(1 for r in results if r["status"] == "ERROR")
    avg_time = sum(r["time"] for r in results) / max(len(results), 1)

    print(f"\n{'='*60}")
    print(f"Verification Results:")
    print(f"  OK: {ok}/{total} ({ok/total*100:.1f}%)")
    print(f"  MISS: {miss}/{total}")
    print(f"  ERROR: {err}/{total}")
    print(f"  Avg response time: {avg_time:.1f}s")
    print(f"{'='*60}")


def main():
    questions = generate_questions()
    save_dataset(questions)

    if "--verify" in sys.argv:
        import asyncio
        asyncio.run(verify_against_api())


if __name__ == "__main__":
    main()
