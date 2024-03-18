from transformers import AutoTokenizer, AutoModelForCausalLM

# 저장된 모델 경로 설정
model_path = "D:/Project/gemma2b/models--beomi--gemma-ko-7b/snapshots/9c24b9c4ab362ca141e8dc1c8fb9cb124c1a136e"

print("start")
# 토크나이저와 모델 로드
tokenizer = AutoTokenizer.from_pretrained(model_path)
print("tokenizer 완료")

model = AutoModelForCausalLM.from_pretrained(model_path)
print("model 로드 완료")

# 모델 사용 예시
input_text = "안녕하세요, 이 모델은 한글 텍스트 생성에 사용됩니다."
input_ids = tokenizer(input_text, return_tensors="pt")["input_ids"]
outputs = model.generate(input_ids=input_ids, max_length=50, num_return_sequences=3, do_sample=True)

for output in outputs:
    print(tokenizer.decode(output, skip_special_tokens=True))