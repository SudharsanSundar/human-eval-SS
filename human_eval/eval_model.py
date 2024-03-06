from models import GPT, TogModel
from human_eval.data import read_problems, write_jsonl
import tqdm

import pprint as pp
from pprint import PrettyPrinter

ppr = pp.PrettyPrinter()

prompt_database = []

def yield_items(answers: dict) -> None:
    for answer_key in answers:
        answer_dict = {}
        answer_dict['task_id'] = answer_key
        answer_dict['completion'] = answers[answer_key]
        yield answer_dict


def generate_answers(model, questions, out_file_path_prefix='gpt3.5_codeOnlyPrompt') -> dict:
    answers = {}

    counter = 0

    for question in questions:
        instruction = '''Please generate the code
                        without any comments, explanations, test cases, or additional text. 
                        Output should be the code only.   '''
        code_prompt = questions[question]['prompt']
        # prompt = instruction + '\n' + code_prompt
        prompt = instruction + code_prompt
        answer = model.answer_txt(prompt=prompt)

        answers[question] = answer

        print('- - - QUESTION', question, '\n', answer)

        
        counter += 1
        if counter == 3:
          break

    folder_prefix = '../data/'
    out_file_path = folder_prefix + out_file_path_prefix + '_HumanEval_All_Results.jsonl'
    write_jsonl(out_file_path, tqdm.tqdm(yield_items(answers)))

    return answers

def generate_answers_with_test_cases(model, questions, out_file_path_prefix='gpt3.5_codeOnlyPrompt', ) -> dict:
    answers = {}

    counter = 0

    for question in questions:
        instruction = '''Please generate the code
                        without any comments, explanations, test cases, or additional text. 
                        Output should be the code only.   
                        
                        Below are the test cases you need to pass for this task, where candidate 
                        is replaced with the function name ''' + questions[question]['entry_point']
                         
        code_prompt = questions[question]['prompt']
        # prompt = instruction + '\n' + code_prompt
        
        prompt = instruction + code_prompt + "Test Case: " + questions[question]['test']
        print(f'This is the prompt: ', prompt)
        prompt_database.append(prompt)
        answer = model.answer_txt(prompt=prompt)

        answers[question] = answer

        print('- - - QUESTION', question, '\n', answer)

        counter += 1
        if counter == 2:
           break

    folder_prefix = '../data/'
    out_file_path = folder_prefix + out_file_path_prefix + '_HumanEval_All_Results.jsonl'
    write_jsonl(out_file_path, tqdm.tqdm(yield_items(answers)))

    return answers


def extract_code(answers: list, out_file_path_prefix='gpt3.5_codeOnlyPrompt') -> list:
    clean_answers = {}
    iteration = 0
    for answer_key in answers:
        model_answer = answers[answer_key]
        iteration += 1
        #print(f'THIS IS EXAMPLE {iteration} FOR CLEANED UP VERSION')
        # TODO: extract function code
        # - could even just prompt gpt3p5 to return just the answer and nothing else
        clean_answer = process_outputs(model_answer)

        clean_answers[answer_key] = clean_answer
        

        #print('- - - CLEANED ANSWER', '\n', clean_answer)

    folder_prefix = '../data/'
    #out_file_path = folder_prefix + out_file_path_prefix + '_HumanEval_All_Results_CLEAN.jsonl'
    out_file_path = folder_prefix + out_file_path_prefix + 'jessicasmallex.jsonl'
    write_jsonl(out_file_path, tqdm.tqdm(yield_items(clean_answers)))

    return answers

def process_outputs(elem):
    print("This was elem before: \n", elem)
    start = elem.find("```python")
    length = len("```python")
    end = elem.find("```", start+1)
    if end == -1:
        end = len(elem)
    if start == -1:
        elem = elem[:end]
    if start != -1:
        elem = elem[start+length:end]
    print("This was elem after: \n", elem)
    return elem



def main():
    gpt3p5 = GPT()
    gpt4 = GPT(model='gpt-4-0125-preview')

    problems = read_problems()
    print('- - - -  PROBLEMS')
    ppr.pprint(problems)

    answers = generate_answers(gpt3p5, problems) # removed limit
    #answers = generate_answers_with_test_cases(gpt3p5, problems) 
    print('- - - - ANSWERS')
    ppr.pprint(answers)

    extract_code(answers)

    #print("THESE WERE ALL OF THE PROMPTS: ", prompt_database)


if __name__ == "__main__":
    main()
