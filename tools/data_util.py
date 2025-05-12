import numpy as np
import os
import re

import datasets

def load_gsm_data():
    gsm8k = datasets.load_dataset('gsm8k', 'main')
    train_data = gsm8k['train']
    test_data = gsm8k['test']

    return train_data, test_data

ANS_RE = re.compile(r"#### (\-?[0-9\.\,]+)")
INVALID_ANS = "[invalid]"

SUBSTITUTIONS = [
    ('an ', ''), ('a ', ''), ('.$', '$'), ('\\$', ''), (r'\ ', ''),
    (' ', ''), ('mbox', 'text'), (',\\text{and}', ','),
    ('\\text{and}', ','), ('\\text{m}', '\\text{}')
]

REMOVED_EXPRESSIONS = [
    'square', 'ways', 'integers', 'dollars', 'mph', 'inches', 'ft',
    'hours', 'km', 'units', '\\ldots', 'sue', 'points', 'feet',
    'minutes', 'digits', 'cents', 'degrees', 'cm', 'gm', 'pounds',
    'meters', 'meals', 'edges', 'students', 'childrentickets', 'multiples',
    '\\text{s}', '\\text{.}', '\\text{\ns}', '\\text{}^2',
    '\\text{}^3', '\\text{\n}', '\\text{}', r'\mathrm{th}',
    r'^\circ', r'^{\circ}', r'\;', r',\!', '{,}', '"', '\\dots'
]

class CoTReasoning():
    def __init__(self, train_dataset) -> None:
        self.train_dataset = train_dataset
        pass

    def clean_gt_answer(self, answer_str: str):
        cleaned_text = re.sub('<<.*?>>', '', answer_str)
        return cleaned_text

    def extract_answer(self, completion):
        match = ANS_RE.search(completion)
        if match:
            match_str = match.group(1).strip()
            match_str = match_str.replace(",", "")
            return match_str
        else:
            return INVALID_ANS

    def format_question(self, data_instance):
        question = data_instance['question']
        answer = data_instance['answer']
        cleaned_answer = self.clean_gt_answer(answer)
        answer_lines = cleaned_answer.split('\n')
        cots = '\n'.join(answer_lines[:-1])
        re.compile(r"#### (\-?[0-9\.\,]+)")
        final_answer = self.extract_answer(answer_lines[-1])

        prompt = "Question: " + question + '\n' + "Let's think step by step\n"
        prompt += cots
        prompt += f'\nThe answer is {final_answer}'
        return prompt


    def refine_answer(self, answer):
        cleaned_answer = self.clean_gt_answer(answer)
        answer_lines = cleaned_answer.strip().split('\n')
        # answer_lines = cleaned_answer.split('\n')
        # print(answer_lines)
        # answer_lines = answer_lines[1:-2]
        cots = '\n'.join(answer_lines[:-1])
        re.compile(r"#### (\-?[0-9\.\,]+)")
        final_answer = self.extract_answer(answer_lines[-1])

        answer = cots
        answer += f'\nThe answer is {final_answer}'

        # print(cots)
        # import ipdb
        # ipdb.set_trace()
        return answer


    def construct_query(self, prompt_list, query):
        whole_cot = '\n\n'.join(prompt_list)
        prompt_q = whole_cot + '\n\nQuestion: ' + query + '\n'
        return prompt_q

    def compute_complexity(self, dataset, return_k = 8):
        answer_list = dataset['answer']
        num_steps = np.array([len(x.split('\n')) - 1 for x in answer_list])
        max_steps = np.max(num_steps)
        top_k_index = np.argsort(-num_steps)[:return_k]
        max_step_index = np.array([x for x in range(len(num_steps)) if num_steps[x] == max_steps])
        return num_steps, top_k_index, max_step_index

    # def self_mistake_evaluation(self, ):
    #     num_steps, top_k_index, max_step_index = self.compute_complexity(self.train_dataset)
    #     retun num_steps, top_k_index, max_step_index

    def test_answer(self, pred_str, ans_str):
        pattern = '\d*\.?\d+'
        pred = re.findall(pattern, pred_str)
        if(len(pred) >= 1):
            # print(pred_str)
            pred = pred[-1]
            gold = re.findall(pattern, ans_str)
            # print(ans_str)
            gold = gold[-1]            
            return pred == gold
        else: return False

    def test_answer_v2(self, pred_str, ans_str):
        ANS_RE = re.compile(r"\$?([0-9,]+)\.?\d*%?")
        pred = re.findall(ANS_RE, pred_str)
        if(len(pred) >= 1):
            pred = pred[-1]
            gold = re.findall(ANS_RE, ans_str)
            gold = gold[-1]            
            pred = pred.replace(",", "").replace(" ", "")
            gold = gold.replace(",", "").replace(" ", "")
            # if pred != gold:
            #     print(pred, gold)
                # pause = input("???")
            return pred == gold
        else: return False
        
    def find_numbers(self, pred_str):
        ANS_RE = re.compile(r"(-?[0-9,]+\.?\d*%?)")
        pred = re.findall(ANS_RE, pred_str)
        if(len(pred) >= 1): 
            for num in pred:
                try: 
                    final = float(num)
                    return num
                except: continue
        else: return INVALID_ANS
        
    def extract_final(self, pred_str):
        ANS_RE = re.compile(r"(-?[0-9,]+\.?\d*%?)")
        pred = re.findall(ANS_RE, pred_str)
        if(len(pred) >= 1): 
            final = pred[-1].replace(",", "")
            if (final.endswith('.')): final = final.replace('.', '')
            return final
        else: return INVALID_ANS
    
    def extract_final_mc2(self, pred_str):
        ANS_RE = re.compile(r"(?<!\w)[AB](?!\w)")
        pred = re.findall(ANS_RE, pred_str)
        if(len(pred) >= 1): return pred[-1]
        else: return INVALID_ANS

    def extract_final_mc3(self, pred_str):
        ANS_RE = re.compile(r"(?<!\w)[ABC](?!\w)")
        pred = re.findall(ANS_RE, pred_str)
        if(len(pred) >= 1): return pred[-1]
        else: return INVALID_ANS
        
    def extract_all_final(self, pred_str):
        ANS_RE = re.compile(r"(?<!\w)[ABC](?!\w)")
        pred = re.findall(ANS_RE, pred_str)
        if(len(pred) >= 1): return pred[-1]
        else: return INVALID_ANS

    def ensure_choice2(self, pred_str):
        ANS_RE = re.compile(r"(?<!\w)(A or B)(?!\w)")
        pred = re.findall(ANS_RE, pred_str)
        if(len(pred) >= 1): return pred[-1]
        else: return INVALID_ANS

    def parse_pred_ans(self, filename):
        with open(filename) as fd: lines = fd.readlines()
        am, a = None, None
        num_q, acc = 0, 0
        current_mode = 'none'
        questions = []
        ans_pred = []
        ans_gold = []
        idx_list = []
        wrong_idx_list = []
        all_idx_list = []
        curr_idx = -1
        for l in lines:
            # if(l.startswith('Question: ')):
            # if(l.startswith('Q: Question: ')):
            # if(l.startswith('Q: ')):
            if(l.startswith('Q: ') or l.startswith('Question: ')):
                if(am is not None and a is not None):
                    questions.append(q)
                    ans_pred.append(am)
                    ans_gold.append(a)
                    if(self.test_answer_v2(am, a)):
                    # if(self.test_answer(am, a)):
                        acc += 1
                    else:
                        # print(am)
                        # print(a)
                        # pause = input("???")
                        wrong_idx_list.append(all_idx_list[-2])
                current_mode = 'q'
                q = l
                num_q += 1
            elif(l.startswith('Options: ')): pass
            elif(l.startswith('A_model:')):
                current_mode = 'am'
                am = l
            elif(l.startswith('A:')):
                current_mode = 'a'
                a = l
            elif (l.startswith("idx:")):
                curr_mode = 'idx'
                curr_idx = int(l[4:])
                all_idx_list.append(curr_idx)
            else:
                # print(l, l=='', l.isspace())
                if(current_mode == 'q'): q += l
                elif(current_mode == 'am'): am += l
                elif(current_mode == 'a'): a += l
                elif l == "\n":continue
                else:
                    # print(l)
                    raise ValueError(current_mode)
                    
        questions.append(q)
        ans_pred.append(am)
        ans_gold.append(a)
        if(self.test_answer_v2(am, a)):
            acc += 1
        else:
            wrong_idx_list.append(all_idx_list[-1])

        print('num_q %d correct %d ratio %.4f' % (num_q, acc, float(acc / num_q)))
        return questions, ans_pred, ans_gold, wrong_idx_list, all_idx_list

    def test_finished(self, ans_model):
        if('answer is' in ans_model): return True
        else: return False

    def extract_ans(self, ans_model):
        ans_model = ans_model.split('\n')
        ans = []
        residual = []
        for li, al in enumerate(ans_model):
            ans.append(al)
            if('answer is' in al):
                break
        residual = list(ans_model[li + 1:])
        ans = '\n'.join(ans)
        residual = '\n'.join(residual)
        return ans, residual
    
    def rmv_newlines(self, pred):
        return pred.replace('\n', ' ').replace('..', '.').replace('  ', ' ')
    
    def normalize_final_answer(self, final_answer: str) -> str:
        """Normalize a final answer to a quantitative reasoning question."""
        # final_answer = final_answer.split('=')[-1]

        for before, after in SUBSTITUTIONS:
            final_answer = final_answer.replace(before, after)
        
        for expr in REMOVED_EXPRESSIONS:
            final_answer = final_answer.replace(expr, '')

        # Extract answer that is in LaTeX math, is bold,
        # is surrounded by a box, etc.
        final_answer = re.sub(r'(.*?)(\$)(.*?)(\$)(.*)', r'$\3$', final_answer)
        final_answer = re.sub(r'(\\text\{)(.*?)(\})', r'\2', final_answer)
        final_answer = re.sub(r'(\\textbf\{)(.*?)(\})', r'\2', final_answer)
        final_answer = re.sub(r'(\\overline\{)(.*?)(\})', r'\2', final_answer)
        final_answer = re.sub(r'(\\boxed\{)(.*)(\})', r'\2', final_answer)

        # Normalize shorthand TeX:
        # \fracab -> \frac{a}{b}
        # \frac{abc}{bef} -> \frac{abc}{bef}
        # \fracabc -> \frac{a}{b}c
        # \sqrta -> \sqrt{a}
        # \sqrtab -> sqrt{a}b
        final_answer = re.sub(r'(frac)([^{])(.)', r'frac{\2}{\3}', final_answer)
        final_answer = re.sub(r'(sqrt)([^{])', r'sqrt{\2}', final_answer)
        final_answer = final_answer.replace('$', '')

        # Normalize 100,000 -> 100000
        if final_answer.replace(',', '').isdigit():
            final_answer = final_answer.replace(',', '')

        return final_answer
    
    def extract_boxed(self, response: str) -> str:
        match = re.search(r'\\boxed(\{.*?\})(?![{|}])', response)
        return match.group(1) if match else INVALID_ANS
    
    def extract_underline(self, response: str) -> str:
        match = re.search(r'\\underline(\{.*?\})(?![{|}])', response)
        return match.group(1) if match else INVALID_ANS
    
    def extract_formatted_ABC(self, response: str, format_start: str, format_end: str) -> str:
        start_of_ans_idx = response.find(format_start) + len(format_start)
        end_of_ans_idx   = start_of_ans_idx + response[start_of_ans_idx:].find(format_end)
        try: return self.extract_final_mc3(response[start_of_ans_idx:end_of_ans_idx])
        except: return None

def majority_vote(answers):
    ans2freq = {}
    max_freq = 0
    max_ans = None
    for ans in answers:
        if ans not in ans2freq: ans2freq[ans] = 1
        else: ans2freq[ans] += 1
        if ans2freq[ans] > max_freq:
            max_ans = ans
    return max_ans, max_freq




