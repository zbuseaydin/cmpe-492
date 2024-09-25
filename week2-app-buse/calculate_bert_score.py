import bert_score
import os

generated_summaries_files = [f"./summaries/{file_name}" for file_name in os.listdir("./summaries") if not file_name.startswith(".")]
original_texts_files = [f"./my-documents/{file_name}" for file_name in os.listdir("./my-documents") if not file_name.startswith(".")]
print(generated_summaries_files)
print(original_texts_files)
generated_summaries = []
for file in generated_summaries_files:
    with open(file, "rb") as f:
        content = ""
        for line in f.readlines():
            content += str(line)
        generated_summaries.append(content)

original_texts = []
for file in original_texts_files:
    with open(file, "rb") as f:
        content = ""
        for line in f.readlines():
            content += str(line)
        original_texts.append(content)
        

# Calculate BERTScore
P, R, F1 = bert_score.score(generated_summaries, original_texts, lang='en', verbose=True)

# Convert to lists
precision = P.tolist()
recall = R.tolist()
f1 = F1.tolist()

# Calculate average scores
average_precision = sum(precision) / len(precision)
average_recall = sum(recall) / len(recall)
average_f1 = sum(f1) / len(f1)

print(f"Average BERTScore Precision: {average_precision:.4f}")
print(f"Average BERTScore Recall: {average_recall:.4f}")
print(f"Average BERTScore F1: {average_f1:.4f}")
