from utils import *

def fast_extract_answer(response) :
    response = response.strip()
    # Direct Strategy Multi-Choice
    # A / A:
    for ch in 'ABCDEFGH':
        if response.upper() == ch or response.startswith(f'{ch}:'):
            return ch

    # Direct Strategy Open-ended
    # 1
    if is_number(response):
        return response

    # CoT strategy
    if 'boxed{' in response:
        try:
            model_answers = extract_full_boxed_content(response)
            if model_answers:
                # for coding
                # \\boxed{\\text{}}
                try:
                    text_content = re.findall(r'\\text{(.*?)}', model_answers[-1])
                    if text_content:
                        return text_content[-1].strip()
                except Exception:
                    pass
                return model_answers[-1].strip()
        except Exception:
            pass
    # for coding
    # the correct answer is\n D.
    for flag in ['final answer is', 'correct answer is', 'answer should be', 'answer is']:
        if flag in response.lower():
            try:
                model_answer = response.lower().split(flag)[-1].strip()
                return model_answer.split('\n')[0].split('.')[0]
            except Exception:
                pass

    return response

text = "Answer is E."
print(fast_extract_answer(text))