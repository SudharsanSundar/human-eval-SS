from models import GPT, TogModel
from human_eval.data import read_problems, write_jsonl, stream_jsonl
import tqdm
import os

import pprint as pp
from pprint import PrettyPrinter

ppr = pp.PrettyPrinter()


def yield_items(answers: dict) -> None:
    for answer_key in answers:
        answer_dict = {'task_id': answer_key, 'completion': answers[answer_key]}
        yield answer_dict


def generate_answers(model, questions, out_fp: str, limit=None) -> dict:
    answers = {}

    counter = 0

    for question in tqdm.tqdm(questions):
        instruction = 'Please complete the function below based on the given function comment.' # maybe add: Please create a function using the given function name, and do not use a lamda function.
        code_prompt = questions[question]['prompt']
        # prompt = instruction + '\n' + code_prompt
        prompt = code_prompt
        answer = model.answer_txt(prompt=prompt)

        answers[question] = answer

        print('- - - QUESTION', question, '\n', answer)

        counter += 1
        if limit and counter == limit:
            break

    write_jsonl(out_fp, tqdm.tqdm(yield_items(answers)))

    return answers


def clean_extract(answer: str) -> str:
    # Gets rid of triple tick marks, "python" header, newlines
    output = answer[10:-4]
    return output


'''
extracts def from model completion and saves it to new file.
'''
def extract_code(answers: dict, out_fp: str) -> dict:
    clean_answers = {}
    extracter = GPT()

    for answer_key in answers:
        model_answer = answers[answer_key]

        extract_prompt = '''Please extract the code for the function within the following text. Extract only the code for the function, and no other code like tests etc. that appear outside the function. Respond with the code for the function, and nothing else.

Text:
{model_completion}'''
        extracted_code = extracter.answer_txt(extract_prompt.format(model_completion=model_answer))

        clean_answer = clean_extract(extracted_code)
        clean_answers[answer_key] = clean_answer

    write_jsonl(out_fp, tqdm.tqdm(yield_items(clean_answers)))

    return answers


def list_generator(lst: list):
    for elem in lst:
        yield elem


'''
Runs model on all problems in given dataset, then extracts the code from the model answers and saves it to a file
'''
def run_inference(model, raw_problems_fp: str, raw_ans_fp: str, clean_ans_fp: str):
    if raw_problems_fp:
        problems = read_problems(evalset_file=raw_problems_fp)
    else:
        problems = read_problems()

    print('- - - - PROBLEMS')
    ppr.pprint(problems)

    answers = generate_answers(model, problems, raw_ans_fp)
    print('- - - - ANSWERS')
    ppr.pprint(answers)

    extract_code(answers, clean_ans_fp)


def run_eval(clean_ans_fp: str, eval_problems_fp: str = None) -> None:
    if eval_problems_fp:
        eval_cmd = 'evaluate_functional_correctness {} --problem_file={}'.format(clean_ans_fp, eval_problems_fp)
    else:
        print('No eval problems file specified, so defaulting to human eval problems')
        eval_cmd = 'evaluate_functional_correctness {}'.format(clean_ans_fp)

    print('Run this command from the root of the repo to eval: ', eval_cmd)


def main():
    gpt3p5 = GPT()
    gpt4 = GPT(model='gpt-4-0125-preview')

    mbpp_raw_problems_fp = '../data/mbpp-problems-raw-INFERENCE.jsonl'
    mbpp_clean_problems_fp = '../data/mbpp-problems-clean-EVAL.jsonl'

    mbpp_raw_answers_fp = '../data/mbpp-gpt4-defaultPrompt-rawAnswers.jsonl'
    mbpp_clean_answers_fp = '../data/mbpp-gpt4-defaultPrompt-cleanAnswers.jsonl'

    run_inference(gpt4, raw_problems_fp=mbpp_raw_problems_fp, raw_ans_fp=mbpp_raw_answers_fp, clean_ans_fp=mbpp_clean_answers_fp)

    run_eval(clean_ans_fp=mbpp_clean_answers_fp, eval_problems_fp=mbpp_clean_problems_fp)
    # evaluate_functional_correctness data/mbpp-gpt4-defaultPrompt-cleanAnswers.jsonl --problem_file=data/mbpp-problems-clean-EVAL.jsonl


if __name__ == "__main__":
    main()
