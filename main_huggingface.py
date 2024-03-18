from transformers import AutoTokenizer, AutoModelForCausalLM

# 모델 이름 설정
model_name = "beomi/open-llama-2-ko-7b"

print("start")

# 토크나이저와 모델 로드
tokenizer = AutoTokenizer.from_pretrained(model_name)

print("tokenizer 완료")

model = AutoModelForCausalLM.from_pretrained(model_name)

print("모델 로드 완료")

# 모델 사용 예시
input_text = "안녕하세요, 이 모델은 한글 텍스트 생성에 사용됩니다."
input_ids = tokenizer(input_text, return_tensors="pt")["input_ids"]
outputs = model.generate(input_ids=input_ids, max_length=50, num_return_sequences=3, do_sample=True)

for output in outputs:
    print(tokenizer.decode(output, skip_special_tokens=True))