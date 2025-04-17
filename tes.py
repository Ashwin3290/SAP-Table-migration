import spacy

def check_distinct_requirement(sentence):
    """
    Analyzes a sentence to determine if it contains words semantically similar to 'distinct' or 'unique',
    which would indicate a need for DISTINCT in SQL queries.
    
    Args:
        sentence (str): The input sentence/query to analyze
        
    Returns:
        bool: True if the sentence likely requires distinct values, False otherwise
    """
    # Load the spaCy model - using the medium English model for better word vectors
    nlp = spacy.load("en_core_web_md")
    
    # Process the input sentence
    doc = nlp(sentence.lower())
    
    # Target words we're looking for similarity to
    target_words = ["distinct", "unique", "different", "individual", "separate"]
    target_docs = [nlp(word) for word in target_words]
    
    similarity_threshold = 0.9
    
    direct_keywords = ["distinct", "unique", "duplicates", "duplicate", "duplicated", "deduplicate", "deduplication"]
    for token in doc:
        if token.text in direct_keywords:
            return True
    
    for token in doc:
        if token.is_stop or token.is_punct:
            continue
        
        # Check similarity with each target word
        for target_doc in target_docs:
            similarity = token.similarity(target_doc[0])
            if similarity > similarity_threshold:
                return True
    
    return False

# Example usage
if __name__ == "__main__":
    test_sentences = [
        "Show me all employees",
        "Get the unique departments in the company",
        "I need a list of distinct product categories",
        "Show me separate entries for each customer",
        "Find all individual transaction types",
        "Count the number of employees in each department",
        "Remove any duplicate records from the results",
        "Show me only one record per customer"
    ]
    
    for sentence in test_sentences:
        result = check_distinct_requirement(sentence)
        print(f"'{sentence}' - Requires DISTINCT: {result}")