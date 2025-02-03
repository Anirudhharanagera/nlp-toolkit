
import torch
from transformers import (
    BertTokenizer,
    BertForQuestionAnswering,
    BartForConditionalGeneration,
    BartTokenizer,
    pipeline,
)
import pandas as pd
from datasets import load_dataset


def load_squad():
    #Load the SQuAD dataset from Hugging Face
    print("Loading SQuAD dataset...")
    dataset = load_dataset("squad", split="validation[:100]")
    return pd.DataFrame(dataset)


def load_cnn_dailymail():
    #Load the CNN/DailyMail dataset from Hugging Face
    print("Loading CNN/DailyMail dataset...")
    dataset = load_dataset("cnn_dailymail", "3.0.0", split="test[:50]")
    return pd.DataFrame(dataset)


def question_answering(squad_df):
    #Perform question answering using a pre-trained BERT model.
    
    print("
--- Question Answering Task ---")

    model_name = "bert-large-uncased-whole-word-masking-finetuned-squad"
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertForQuestionAnswering.from_pretrained(model_name).to("cuda" if torch.cuda.is_available() else "cpu")

    unique_contexts = squad_df["context"].unique()

    while True:
        print("
Available Context Options:
")
        for i, context in enumerate(unique_contexts[:10], start=1):
            print(f"{i}. {context[:300]}...")  # Display first 300 characters

        print("
Type the number of the context you want, 'menu' to go back, or 'exit' to quit.")
        user_choice = input("Enter your choice: ").strip()

        if user_choice.lower() == "exit":
            print("Exiting the Question Answering Task.")
            break
        elif user_choice.lower() == "menu":
            return

        try:
            choice_index = int(user_choice) - 1
            chosen_context = unique_contexts[choice_index]
        except (ValueError, IndexError):
            print("Invalid choice. Please select a valid option.")
            continue

        print(f"
Selected context:
{chosen_context[:500]}...
")

        while True:
            question = input("Enter your question (or type 'menu' to return): ").strip()

            if question.lower() == "menu":
                break

            inputs = tokenizer(question, chosen_context, return_tensors="pt", truncation=True).to(model.device)
            outputs = model(**inputs)
            start_index = torch.argmax(outputs.start_logits)
            end_index = torch.argmax(outputs.end_logits) + 1
            answer = tokenizer.convert_tokens_to_string(
                tokenizer.convert_ids_to_tokens(inputs["input_ids"][0][start_index:end_index])
            )
            print(f"Question: {question}")
            print(f"Answer: {answer}
")
            print("-" * 50)


def text_summarization(cnn_df):
    
    #Perform text summarization using BART model.
    print("
--- Text Summarization Task ---")

    summarizer_model_name = "facebook/bart-large-cnn"
    tokenizer = BartTokenizer.from_pretrained(summarizer_model_name)
    model = BartForConditionalGeneration.from_pretrained(summarizer_model_name).to("cuda" if torch.cuda.is_available() else "cpu")

    articles = cnn_df["article"]

    while True:
        print("
Available Article Options:
")
        for i, article in enumerate(articles[:10], start=1):
            print(f"{i}. {article[:300]}...")

        print("
Type the number of the article to summarize, 'menu' to return, or 'exit' to quit.")
        user_choice = input("Enter your choice: ").strip()

        if user_choice.lower() == "exit":
            print("Exiting the Text Summarization Task.")
            break
        elif user_choice.lower() == "menu":
            return

        try:
            choice_index = int(user_choice) - 1
            selected_article = articles.iloc[choice_index]
        except (ValueError, IndexError):
            print("Invalid choice. Please select a valid option.")
            continue

        print(f"
Selected article:
{selected_article[:500]}...
")

        print("Summarizing the article...
")
        inputs = tokenizer(selected_article, return_tensors="pt", max_length=1024, truncation=True).to(model.device)
        summary_ids = model.generate(inputs["input_ids"], max_length=50, min_length=25, length_penalty=2.0, num_beams=4, early_stopping=True)
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        print(f"Summary: {summary}
")
        print("-" * 50)


def document_comprehension(squad_df):
    #Perform document comprehension using a question-answering pipeline.
    
    print("
--- Document Comprehension Task ---")

    model_name = "bert-large-uncased-whole-word-masking-finetuned-squad"
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertForQuestionAnswering.from_pretrained(model_name).to("cuda" if torch.cuda.is_available() else "cpu")
    nlp = pipeline("question-answering", model=model, tokenizer=tokenizer)

    unique_contexts = squad_df["context"].unique()

    while True:
        print("
Available Context Options:
")
        for i, context in enumerate(unique_contexts[:10], start=1):
            print(f"{i}. {context[:300]}...")

        print("
Type the number of the context you want, 'menu' to go back, or 'exit' to quit.")
        user_choice = input("Enter your choice: ").strip()

        if user_choice.lower() == "exit":
            print("Exiting the Document Comprehension Task.")
            break
        elif user_choice.lower() == "menu":
            return

        try:
            choice_index = int(user_choice) - 1
            chosen_context = unique_contexts[choice_index]
        except (ValueError, IndexError):
            print("Invalid choice. Please select a valid option.")
            continue

        print(f"
Selected context:
{chosen_context[:500]}...
")

        while True:
            command = input("Type your question about the context (or 'menu' to return): ").strip()

            if command.lower() == "menu":
                break

            response = nlp(question=command, context=chosen_context)
            print(f"
Question: {command}")
            print(f"Answer: {response['answer']}
")
            print("-" * 50)


def main():
    
    # Main function to execute tasks interactively.
    print("Welcome to the Transformer-based NLP Toolkit")
    print("Loading datasets, please wait...
")
    squad_df = load_squad()
    cnn_df = load_cnn_dailymail()

    while True:
        print("
Select a task to perform:")
        print("1. Question Answering")
        print("2. Text Summarization")
        print("3. Document Comprehension")
        print("4. Exit")

        choice = input("Enter your choice (1/2/3/4): ")

        if choice == "1":
            question_answering(squad_df)
        elif choice == "2":
            text_summarization(cnn_df)
        elif choice == "3":
            document_comprehension(squad_df)
        elif choice == "4":
            print("Exiting the program. Goodbye!")
            break
        else:
            print("Invalid choice. Please enter a number between 1 and 4.")


if __name__ == "__main__":
    main()
