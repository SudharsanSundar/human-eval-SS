from models import GPT, TogModel
from human_eval.data import read_problems, write_jsonl, stream_jsonl
import tqdm

import pprint as pp
from pprint import PrettyPrinter

ppr = pp.PrettyPrinter()


def list_generator(lst: list):
    for elem in lst:
        yield elem


def find_errors(results_fp):
    file = stream_jsonl(results_fp)
    errors = {}
    for line in file:
        result = line['result']
        if 'failed' in result:
            if result in errors:
                errors[result] += 1
            else:
                errors[result] = 1

    sorted_errs = sorted(errors.items(), key=lambda x: x[1], reverse=True)
    ppr.pprint(sorted_errs)

    total_non_logic_errs = 0
    total_name_errs = 0
    for entry in sorted_errs:
        if entry[0] != 'failed: ':
            total_non_logic_errs += entry[1]
            if 'name ' in entry[0] and ' is not defined' in entry[0]:
                total_name_errs += 1

    print('non logic errs', total_non_logic_errs, 'num name errs of that', total_name_errs)


def find_all_imports(raw_answers):
    file = stream_jsonl(raw_answers)

    imports = set()
    for line in file:
        for code_line in line['completion'].split('\n'):
            print(code_line)
            if 'import' in code_line:
                imports.add(code_line)

    ppr.pprint(imports)
    print(repr('\n'.join(imports)))


def main():
    gpt3p5 = GPT()
    gpt4 = GPT(model='gpt-4-0125-preview')

    mbpp_raw_problems_fp = '../data/mbpp_problems_raw.jsonl'
    mbpp_partially_cleaned_fp = '../data/mbpp_problems_clean.jsonl'
    mbpp_answers_raw_fp = '../data/gpt3.5_defaultPrompt_MBPP_All_Results.jsonl'
    mbpp_answers = '../data/gpt3.5_defaultPrompt_MBPP_All_Results_CLEAN2.jsonl'
    mbpp_results_fp = '../data/gpt3.5_defaultPrompt_MBPP_All_Results_CLEAN2.jsonl_results.jsonl'

    # NOTE: You should not have to run things in this file. Plz check w me/double check before running these to make sure you don't overwrite important things.


if __name__ == "__main__":
    main()
