from pathlib import Path
from openai import OpenAI
import json
import argparse
from tqdm import tqdm
import time
import re

if __name__ == '__main__':

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--prompts', type=str, default='./prompts/label_q_eval')
    argparser.add_argument('--out_path', type=str, default='results/')
    # 실행할 때 --eval과 key는 반드시 지정해줘야 한다. 나머지는 default.
    argparser.add_argument('--eval', type=str, default='data/label_anchored_q.json')
    argparser.add_argument('--key', type=str, required=True)
    argparser.add_argument('--model', type=str, default='gpt-5-mini')
    args = argparser.parse_args()

    with open(args.eval, 'r', encoding='utf-8') as f:
        q_set = f.read()
    
    target_dir = Path(args.prompts)
    txt_files = list(target_dir.glob('*.txt'))

    client = OpenAI(api_key=args.key)
    for file_path in tqdm(txt_files, desc="Evaluating..."):
        with open(file_path, 'r', encoding='utf-8') as f:
            prompt = f.read()
            cur_prompt = prompt.replace('{{question_set}}', q_set)
            while True:
                    try:
                        _response = client.chat.completions.create(
                            model=args.model,
                            messages=[
                                {"role": "system", "content": "You are a cold, robotic evaluator. Output ONLY the number, no prose, no explanation."},
                                {"role": "user", "content": cur_prompt}],
                            temperature=1,
                            # 이 모델에서는 temperature가 1만 지원됨.
                            max_completion_tokens=10000,
                            n=8
                        )
                        time.sleep(0.5)

                        # 결과 파싱 (객체 접근 방식)
                        all_responses = [choice.message.content.strip() for choice in _response.choices]
                        print(all_responses)
                        
                        # [디버깅 추가] 모델이 뭐라고 하는지 직접 확인
                        print(f"\n[DEBUG] 첫 번째 응답: {all_responses[0]}")

                        # 기댓값 계산 (숫자만 추출)
                        scores = []
                        for res in all_responses:
                            # 문자열에서 숫자를 모두 찾습니다.
                            nums = re.findall(r'[1-5]', res)
                            if nums:
                                # 모델이 수다를 떨다가 마지막에 "최종 점수는 5"라고 할 확률이 높으므로
                                # 가장 마지막에 등장하는 숫자를 가져옵니다.
                                scores.append(int(nums[-1]))

                        avg_score = sum(scores) / len(scores) if scores else 0
                        print(avg_score)

                        # 결과 데이터 구성
                        eval_data = {
                            "metric": file_path.stem,
                            "avg_score": avg_score,
                            "responses": all_responses
                        }

                        result = []
                        result.append(eval_data)

                        # 파일 저장 (파일명만 stem으로 추출)
                        result_path = Path(args.out_path) / f"{Path(args.eval).stem}_{file_path.stem}.json"
                        with open(result_path, 'w', encoding='utf-8') as f:
                            json.dump(result, f, indent=4, ensure_ascii=False)

                        time.sleep(0.5)
                        break  # 성공 시 while 루프 탈출

                    except Exception as e:
                        print(f"\nError occurred: {e}. Retrying in 2 seconds...")
                        time.sleep(2)
