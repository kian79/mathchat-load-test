import json
import argparse
import csv

def extract_qa(input_path, questions_output, qa_output, max_items):
    questions = []
    qa_pairs = []

    with open(input_path, 'r', encoding='utf-8') as f:
        count = 0
        for line in f:
            if count >= max_items:
                break
            if not line.strip():
                continue
            try:
                entry = json.loads(line)
                question = entry.get('question', '').strip()
                answer = entry.get('answer', '').strip()

                if not question or not answer:
                    continue

                # Extract final answer after '####'
                ground_truth = answer.split('####')[-1].strip()

                questions.append({'prompt': question})
                qa_pairs.append({'question': question, 'answer': ground_truth})
                count += 1
            except json.JSONDecodeError:
                continue

    # Write questions to CSV
    with open(questions_output, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=['prompt'])
        writer.writeheader()
        writer.writerows(questions)

    # Write QA pairs to JSONL
    with open(qa_output, 'w', encoding='utf-8') as qa_out:
        for pair in qa_pairs:
            qa_out.write(json.dumps(pair, ensure_ascii=False) + '\n')

    print(f"✅ Extracted {len(questions)} questions to CSV.")
    print(f"✅ Extracted {len(qa_pairs)} question-answer pairs to JSONL.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract first-round questions and ground truth answers.")
    parser.add_argument('--input', type=str, required=True, help="Path to input JSONL file")
    parser.add_argument('--questions_output', type=str, required=True, help="Path to output CSV file for questions")
    parser.add_argument('--qa_output', type=str, required=True, help="Path to output JSONL file for QA pairs")
    parser.add_argument('--max', type=int, default=100, help="Maximum number of entries to extract")

    args = parser.parse_args()
    extract_qa(args.input, args.questions_output, args.qa_output, args.max)
