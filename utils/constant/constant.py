# API 키 패턴과 모델 패턴 정의
API_KEY_PATTERN = r"sk-*"
# MODEL_PATTERN = r"gpt-*"

# 사용 가능한 AI 모델 목록
# 첫번재 모델이 openai
#  두번째 모델이 claude
AI_MODEL = [
    "선택해주세요",
    "gpt-4o-mini-2024-07-18",
    "claude-3-sonnet-20240229",
]

AI_PRICING_PER_MILLION_TOKENS = {
    AI_MODEL[1]: 2.50,
    AI_MODEL[2]: 3,
}
