from models import GPT, TogModel
from human_eval.data import read_problems, write_jsonl
import tqdm

import pprint as pp
from pprint import PrettyPrinter

ppr = pp.PrettyPrinter()


def yield_items(answers: dict) -> None:
    for answer_key in answers:
        answer_dict = {}
        answer_dict['task_id'] = answer_key
        answer_dict['completion'] = answers[answer_key]
        yield answer_dict


def generate_answers(model, questions, out_file_path_prefix='gpt3.5_codeOnlyPrompt', limit=10) -> dict:
    answers = {}

    counter = 0

    for question in questions:
        instruction = 'Please complete the function below based on the given function comment.'
        code_prompt = questions[question]['prompt']
        # prompt = instruction + '\n' + code_prompt
        prompt = code_prompt
        answer = model.answer_txt(prompt=prompt)

        answers[question] = answer

        print('- - - QUESTION', question, '\n', answer)

        counter += 1
        if counter == limit:
            break

    folder_prefix = '../data/'
    out_file_path = folder_prefix + out_file_path_prefix + '_HumanEval_All_Results.jsonl'
    write_jsonl(out_file_path, tqdm.tqdm(yield_items(answers)))

    return answers


def extract_code(answers: list, out_file_path_prefix='gpt3.5_codeOnlyPrompt') -> list:
    clean_answers = {}

    for answer_key in answers:
        model_answer = answers[answer_key]

        # TODO: extract function code
        # - could even just prompt gpt3p5 to return just the answer and nothing else
        clean_answer = None

        clean_answers[answer_key] = clean_answer

        print('- - - CLEANED ANSWER', '\n', clean_answer)

    folder_prefix = '../data/'
    out_file_path = folder_prefix + out_file_path_prefix + '_HumanEval_All_Results_CLEAN.jsonl'
    write_jsonl(out_file_path, tqdm.tqdm(yield_items(clean_answers)))

    return answers


def main():
    gpt3p5 = GPT()
    gpt4 = GPT(model='gpt-4-0125-preview')

    problems = read_problems()
    print('- - - -  PROBLEMS')
    ppr.pprint(problems)

    answers = generate_answers(gpt3p5, problems, limit=3)
    print('- - - - ANSWERS')
    ppr.pprint(answers)

    extract_code(answers)


if __name__ == "__main__":
    main()
