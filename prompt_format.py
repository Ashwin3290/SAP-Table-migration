"""
Prompt templates for TableLLM
"""

SINGLE_TABLE_TEMPLATE = """
I need ONLY Python code - DO NOT include any explanations, markdown, or comments outside the code.

CSV Header and First Rows:
{csv_data}

Question: {question}

I'll place your code inside this function template:

def analyze_data(df):
    # CODE STARTS HERE
    
    # CODE ENDS HERE, must set 'result' variable
    return result

REQUIREMENTS:
1. Output ONLY the code that goes between the comments
2. Assign your final output to a variable named 'result'
3. For visualizations use: result = plt.gcf()
4. Handle data errors and empty dataframes
5. NEVER use print() statements
6. NEVER reference file paths
7. Include comments inside your code
"""

DOUBLE_TABLE_TEMPLATE_PRE = """
I need ONLY Restructured user question - DO NOT include any explanations, markdown, or comments outside the code. 

Consider the following to format the question:
1. CSV 1 Header and First rows:{csv_data1}
2. CSV 2 Header and First rows:{csv_data2}

Question : {question}

Resolve the columns of the prompt given from the given table descriptions.
{table_desc}

Your Result should be in the following format:
1. Resolved Column names that will be used in the questions.
2. Structured question with resolved columns.

Final Result should only be the structured prompt and NOTHING else"""

DOUBLE_TABLE_TEMPLATE = """
I need ONLY Python code - DO NOT include any explanations, markdown, or comments outside the code.

CSV 1 Header and First Rows:
{csv_data1}

CSV 2 Header and First Rows:
{csv_data2}

Question: {question}

I'll place your code inside this function template:

def analyze_data(df1, df2):
    # CODE STARTS HERE
    
    # CODE ENDS HERE, must set 'result' variable
    return result

REQUIREMENTS:
1. Output ONLY the code that goes between the comments
2. Assign your final output to a variable named 'result'
3. For visualizations use: result = plt.gcf()
4. Handle data errors and empty dataframes
5. NEVER use print() statements
6. NEVER reference file paths
7. Include comments inside your code
"""

QA_TEMPLATE = """
You are a data analysis assistant. Please answer the following question about this tabular data:

Data description:
{table_descriptions}

Table data:
{table_in_csv}

Question: {question}

Provide a clear and concise answer based only on the data provided.
"""