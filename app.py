from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import os
import uuid
import base64
import pandas as pd
import torch
from dotenv import load_dotenv
from langchain_community.document_loaders import DataFrameLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.chains import LLMChain
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import difflib
import json
import google.generativeai as genai
import numpy as np
import re
# Load environment variables
load_dotenv()

# Global variables
app = Flask(__name__)
CORS(app)
favourites = []

# Configuration
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
DATA_PATH = os.getenv("DATA_PATH", "billing data.csv")
CHROMA_DIR = "chroma_db"
TEMP_DIR = "temp_audio"

# Domain keywords and aliases
BILLING_KEYWORDS = {
    "billing", "revenue", "amount", "pat count", "billtype", "regnumber", "ipno", "patientname",
    "dept", "admitdoctor", "ledger", "groupname", "headername", "orderingdoctor", "orderingdept", 
    "servicename", "time", "billno", "servqty", "taxamount", "concamt", "city", "cityid", 
    "district", "state", "serviceid", "iheader_id", "igroup_id", "iadmit_doc_id", "ord_doc_id", 
    "ideptid", "ord_dept_id", "patid", "opid", "ipid", "pricelistid", "mrd", "userid", 
    "pat_type_id", "patient type", "rate", "corperate type", "corp_type_id", "bedno", "ward", 
    "dynamic amount range", "bedtype", "financialyear", "qtr", "quarter", "avgrpb", "discount", 
    "docconcamt", "net", "bed category", "standard financial year", "date level label", 
    "weekday", "dr", "rev", "city names", "last year", "last month"
}

COLUMN_ALIASES = {
    "dept": "Admitting Department",
    "department": "Admitting Department",
    "admitting department": "Admitting Department",
    "doctor": "Doctor Name",
    "doc": "Doctor Name",
    "admit doctor": "Doctor Name",
    "ordering doctor": "Ordering Doctor",
    "ord doctor": "Ordering Doctor",
    "service": "Service",
    "group": "Group",
    "header": "Header",
    "city": "Cities",
    "cities": "Cities",
    "year": "Year",
    "month": "Month",
    "quarter": "Qtr",
    "qtr": "Qtr",
    "financial year": "Financial Year",
    "fy": "Financial Year",
    "revenue": "Revenue",
    "rev": "Revenue",
    "re": "Revenue",
}

ALIASES = {

    # Cities

    "cbe": "coimbatore",

    "kovai":"coimbatore",

    "cbre":"coimbatore",

    "chn": "chennai",
    "cn": "chennai",

    "chnni": "chennai",

    "coimbtore": "coimbatore",

    "kovai":"coimbatore",
    "cbtre":"coimbatore",
    "chenni": "chennai",

    "mumbai": "mumbai",

    "delhi": "delhi",

    "bangalore": "bengaluru",
    "bng": "bengaluru",
    "beng": "bengaluru",
    "bg":"bengaluru",

    "hyd": "hyderabad",

    "pune": "pune",

    "kolkata": "kolkata",

    "ahmedabad": "ahmedabad",

    "surat": "surat",

    "jaipur": "jaipur",

    "bhopal": "bhopal",

    "indore": "indore",

    "vadodara": "vadodara",

    "cochin": "kochi",

    "trivandrum": "thiruvananthapuram",

    # Departments

    "dermo": "dermatology",

    "dermat": "dermatology",

    "gen med": "general medicine",

    "gen": "general medicine",

    "psych": "psychiatry",

    "uro": "urology",

    "cardio": "cardiology",

    "neuro": "neurology",

    "nerve department" : "neurology",

    "ortho": "orthopedics",

    "pedia": "pediatrics",

    "gyne": "gynecology",

    "ent": "ent",

    "opthal": "ophthalmology",

    "radio": "radiology",

    "gastro": "gastroenterology",

    "gestro": "gastroenterology",

    "gastrology": "gastroenterology",
    "gas":"gastroenterology",

    "nephro": "nephrology",

    "hema": "hematology",

    "onco": "oncology",

    "endo": "endocrinology",

    "pulmo": "pulmonology",

    "rheum": "rheumatology",

    "infect": "infectious diseases",

    "emerg": "emergency medicine",

    "int med": "internal medicine",

    "fam med": "family medicine",

    "sports med": "sports medicine",

    "pain": "pain management",

    "sleep": "sleep medicine",

    "addict": "addiction medicine",

    "redio":"radiology",

    # Services

    "lab": "laboratory",

    "xray": "x-ray",

    "ct": "ct scan",

    "mri": "mri",

    "ultra": "ultrasound",

    # Common misspellings

    "dermotlogy": "dermatology",

    "pscyh": "psychiatry",

    "urolojy": "urology",

    "cardiolgy": "cardiology",

    "neurolgy": "neurology",

    "orthopedcs": "orthopedics",

    "pediatr": "pediatrics",

    "gynecolgy": "gynecology",

    "opthalmology": "ophthalmology",

    "radiolog": "radiology",

    "coimbtor": "coimbatore",

    "chnnai": "chennai",

    "mumbi": "mumbai",

    "delhy": "delhi",

    "bangalor": "bengaluru",

    "hyderbad": "hyderabad",

    # Months

    "jan": "January",

    "feb": "February",

    "mar": "March",

    "apr": "April",

    "may": "May",

    "jun": "June",

    "jul": "July",

    "aug": "August",

    "sep": "September",

    "oct": "October",

    "nov": "November",

    "dec": "December",

    # Quarters

    "q1": "Q1",

    "q2": "Q2",

    "q3": "Q3",

    "q4": "Q4",

    "quarter1": "Q1",

    "quarter2": "Q2",

    "quarter3": "Q3",

    "quarter4": "Q4",

    # Years

    "2023": "2023",

    "2024": "2024",

    "2025": "2025",

    # Other common terms

    "rev": "revenue",

    "amt": "amount",

    "tot": "total",

    "sum": "total",

    # More service abbreviations

    "xray": "x-ray",

    "xr": "x-ray",

    "ctscan": "ct scan",

    "ct": "ct scan",

    "mri": "mri",

    "ultra": "ultrasound",

    "usg": "ultrasound",

    "echo": "echocardiography",

    "ecg": "electrocardiogram",

    "ekg": "electrocardiogram",

    "blood test": "blood investigation",

    "urine test": "urine analysis",

    "stool test": "stool analysis",

    # More department abbreviations

    "med": "general medicine",

    "surg": "general surgery",

    "ortho": "orthopedics",

    "neuro": "neurology",

    "cardio": "cardiology",

    "pulm": "pulmonology",

    "gastro": "gastroenterology",

    "neph": "nephrology",

    "hem": "hematology",

    "onco": "oncology",

    "endo": "endocrinology",

    "rheum": "rheumatology",

    "derm": "dermatology",

    "psych": "psychiatry",

    "pedia": "pediatrics",

    "obs": "obstetrics",

    "gyn": "gynecology",

    "ent": "ent",

    "opthal": "ophthalmology",

    "radio": "radiology",

    # Doctor names - common short forms
    "ananya": "Dr. Ananya Rao",
    "rao": "Dr. Ananya Rao",
    "arjun": "Dr. Arjun Mehta",
    "mehta": "Dr. Arjun Mehta",
    "ava": "Dr. Ava Thomas",
    "thomas": "Dr. Ava Thomas",
    "daniel": "Dr. Daniel Taylor",
    "taylor": "Dr. Daniel Taylor",
    "emily": "Dr. Emily Davis",
    "davis": "Dr. Emily Davis",
    "ethan": "Dr. Ethan Martin",
    "martin": "Dr. Ethan Martin",
    "isabella": "Dr. Isabella Garcia",
    "garcia": "Dr. Isabella Garcia",
    "john": "Dr. John Smith",
    "smith": "Dr. John Smith",
    "karthik": "Dr. Karthik Reddy",
    "reddy": "Dr. Karthik Reddy",
    "michael": "Dr. Michael Brown",
    "brown": "Dr. Michael Brown",
    "neha": "Dr. Neha Kapoor",
    "kapoor": "Dr. Neha Kapoor",
    "olivia": "Dr. Olivia Johnson",
    "johnson": "Dr. Olivia Johnson",
    "priya": "Dr. Priya Sharma",
    "sharma": "Dr. Priya Sharma",
    "rahul": "Dr. Rahul Menon",
    "menon": "Dr. Rahul Menon",
    "rohan": "Dr. Rohan Nair",
    "nair": "Dr. Rohan Nair",
    "sneha": "Dr. Sneha Iyer",
    "iyer": "Dr. Sneha Iyer",
    "sophia": "Dr. Sophia Wilson",
    "wilson": "Dr. Sophia Wilson",
    "william": "Dr. William White",
    "white": "Dr. William White"

}

# Global variables for initialized components
df = None
vectorstore = None
qa_chain = None
embeddings = None
DISTINCT_VALUES = {}


def initialize_gemini():
    """Initialize Gemini API configuration."""
    try:
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        print("Gemini API configured successfully!")
    except Exception as e:
        print(f"Error configuring Gemini API: {e}")


def create_temp_directory():
    """Create temporary directory for audio files."""
    if not os.path.exists(TEMP_DIR):
        os.makedirs(TEMP_DIR)


def load_dataset():
    """Load and validate the dataset."""
    global df, DISTINCT_VALUES
    
    if not os.path.exists(DATA_PATH):
        print("Dataset not found.")
        exit(1)
    
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded dataset with {len(df)} rows and {len(df.columns)} columns")
    
    # Generate distinct values
    for col in df.columns:
        distinct = df[col].dropna().unique().tolist()
        DISTINCT_VALUES[col] = distinct


def initialize_embeddings():
    """Initialize HuggingFace embeddings model."""
    global embeddings
    
    try:
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        print("Embeddings model loaded successfully!")
        return embeddings
    except Exception as e:
        print(f"Could not load embeddings model: {e}")
        exit(1)


def create_documents():
    """Create LangChain documents from DataFrame without text splitting."""
    global df
    
    # Convert rows into LangChain Documents
    loader = DataFrameLoader(df, page_content_column="Admitting Department")
    docs = loader.load()
    print(f"Created {len(docs)} documents from DataFrame")
    
    # Create comprehensive searchable content by including ALL columns
    for i, row in df.iterrows():
        content_parts = []
        for col in df.columns:
            if pd.notna(row[col]):
                if col == 'Revenue':
                    content_parts.append(f"{col}: ₹{row[col]/100000:,.2f} lakhs")
                else:
                    content_parts.append(f"{col}: {row[col]}")
        
        content = ". ".join(content_parts)
        docs[i].page_content = content
        docs[i].metadata = row.to_dict()
    
    return docs

def initialize_vectorstore():
    """Initialize or load the Chroma vectorstore safely."""
    #  Declare all globals FIRST
    global vectorstore, embeddings, df  

    print("Building documents from dataset...")

    # Ensure page content column exists
    page_column = "Admitting Department" if "Admitting Department" in df.columns else df.columns[0]

    # Load the DataFrame as documents
    loader = DataFrameLoader(df, page_content_column=page_column)
    docs = loader.load()

    from langchain.schema import Document

    #  Clean and format each document properly
    clean_docs = []
    for i, row in df.iterrows():
        content_parts = [f"{col}: {row[col]}" for col in df.columns if pd.notna(row[col])]
        content = ". ".join(content_parts)
        clean_docs.append(Document(page_content=str(content), metadata=row.to_dict()))

    # Handle the vectorstore creation/loading
    if os.path.exists(CHROMA_DIR):
        print("Loading existing vectorstore...")
        vectorstore = Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)
    else:
        print(f"Creating new vectorstore with {len(clean_docs)} documents...")
        vectorstore = Chroma.from_documents(clean_docs, embedding=embeddings, persist_directory=CHROMA_DIR)
        vectorstore.persist()
        print("Vectorstore created and saved.")

    return vectorstore


def initialize_llm_chain():
    """Initialize the conversational RAG chain."""
    global vectorstore, qa_chain
    
    if not GROQ_API_KEY:
        print("GROQ_API_KEY not set")
        exit(1)
    
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 50}
    )
    
    llm = ChatGroq(
        model="openai/gpt-oss-120b",
        groq_api_key=GROQ_API_KEY,
    )

    template = """You are a specialized medical billing data assistant with knowledge of all distinct values in the dataset. Only answer questions related to billing data.

Context: {context}

Dataset Information:
- Columns: Year, Qtr, Month, Day, Financial Year, Qtr, Admitting Department, Doctor Name, Ordering Doctor, Service, Group, Header, Cities, Revenue
- Distinct values are available for all columns to provide accurate answers

Question: {question}

Instructions:
- Know all distinct values from all 14 columns (Departments, Doctors, Cities, Services, etc.).
- Respond naturally like ChatGPT.
- Identify if the spelling is wrong, suggest other possible meanings, and recognize partial words.
- For questions about available options, list the relevant distinct values.
- Handle short forms, abbreviations, and spelling mistakes intelligently.
- Stay focused on billing data only; do not provide unrelated explanations.
- Be comprehensive, helpful, and concise.
- If the input cannot be matched, suggest the closest possible matches from the dataset.
- Do not add hallucinations or unnecessary explanations.
- For revenue-related questions, refer to the provided data table for accurate figures.
- Never return plain text; always use a table structure.

Answer:"""

    QA_PROMPT = PromptTemplate(
        template=template,
        input_variables=["context", "question"]
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": QA_PROMPT}
    )
    
    return qa_chain


def is_in_domain(text):
    """Check if the text contains any of the required billing keywords."""
    text_lower = text.lower()
    return any(keyword in text_lower for keyword in BILLING_KEYWORDS)


def normalize_query(query, aliases):
    """Normalize user query using LLM."""
    prompt = PromptTemplate(
        input_variables=["query", "ALIASES"],
        template="""
You are a query normalizer for a hospital revenue dataset.

The user may give short, messy, or misspelled inputs
(city abbreviations, doctor names, department names, months, years, etc).

IMPORTANT: Do NOT change or "correct" words that are likely to be specific group names, header names, or service names from the dataset. Words like "inpatient", "therapy", "surgical", "outpatient", "lab", "operation theatre", etc. should be preserved as-is.

instructions:
- only provide the normalized query alone dont give any explanations or suggestions etc
- provide valid query, double check the spelling and the query
Your job:
1. Correct spelling mistakes (e.g., "revene" → "Revenue", "crdio" → "Cardiology", "coim" -> "coimbatore").
2. Expand abbreviations (e.g., "cbe" → "Coimbatore", "chn" → "Chennai").
3. also add revenue to it if the query doesnt have
4. PRESERVE group names, header names, and service names as they appear in the query
if needed refer the {ALIASES}
Return ONLY the normalized sentence, nothing else.

User query: {query}
"""
    )
    
    llm = ChatGroq(model="openai/gpt-oss-20b")
    normalize_chain = LLMChain(llm=llm, prompt=prompt)
    
    try:
        result = normalize_chain.run(query=query, ALIASES=aliases)
        print(f"normalized_query: {result}")
        return result
    except Exception as e:
        print("Normalization failed:", e)
        return query



import re
import pandas as pd
from datetime import datetime
import dateparser

# Assume df is your DataFrame loaded globally
# Example: df = pd.read_csv("revenue_data.csv")

# Ensure your Date column is properly parsed once when loading:
# df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
# df['Day'] = df['Date'].dt.day
# df['Month'] = df['Date'].dt.month_name()
# df['Year'] = df['Date'].dt.year
# df['Hour'] = df['Date'].dt.hour


def extract_date_filters(question):
    """
    Detects date, month, year, and time ranges in a user query.
    Returns a dictionary with parsed filters.
    """
    q_lower = question.lower()
    filters = {}

    # Parse full natural language date if present
    parsed_date = dateparser.parse(q_lower, settings={'DATE_ORDER': 'DMY'})
    if parsed_date:
        filters['date'] = parsed_date.date()

    # Detect year
    year_match = re.search(r'(20\d{2})', q_lower)
    if year_match:
        filters['year'] = int(year_match.group(1))

    # Detect month names (multiple months)
    months_found = []
    for m in [
        "january","february","march","april","may","june",
        "july","august","september","october","november","december"
    ]:
        if m in q_lower:
            months_found.append(m.title())

    if months_found:
        # Sort months chronologically
        month_order = ['January', 'February', 'March', 'April', 'May', 'June',
                      'July', 'August', 'September', 'October', 'November', 'December']
        filters['months'] = sorted(months_found, key=lambda x: month_order.index(x))
        # For backward compatibility, set first month as 'month'
        filters['month'] = filters['months'][0]

    # Detect quarter (Q1, Q2, etc.)
    quarter_match = re.search(r'\bq([1-4])\b', q_lower)
    if quarter_match:
        filters['quarter'] = f"Q{quarter_match.group(1)}"

    # Detect specific day number
    day_match = re.search(r'\b([0-3]?\d)(?:st|nd|rd|th)?\b', q_lower)
    if day_match:
        filters['day'] = int(day_match.group(1))

    # Detect financial year (e.g., FY2024-2025)
    fy_match = re.search(r'\bfy\s*(\d{4})\s*-\s*(\d{4})\b', q_lower)
    if fy_match:
        filters['financial_year'] = f"FY{fy_match.group(1)}-{fy_match.group(2)}"

    # Detect time range like "9am to 5pm"
    time_range = re.search(r'(\d{1,2}(?:[:.]\d{2})?\s*(?:am|pm)?)\s*(?:to|-)\s*(\d{1,2}(?:[:.]\d{2})?\s*(?:am|pm)?)', q_lower)
    if time_range:
        filters['start_time'] = dateparser.parse(time_range.group(1)).time()
        filters['end_time'] = dateparser.parse(time_range.group(2)).time()

    return filters


def handle_comparison_query(question, filters):
    """Handle comparison queries like 'compare revenue between chennai and mumbai'"""
    global df
    q_lower = question.lower()

    # Apply date filters first
    df_filtered = df.copy()
    if 'date' in filters and 'Date' in df_filtered.columns:
        df_filtered = df_filtered[df_filtered['Date'].dt.date == filters['date']]
    if 'month' in filters and 'Month' in df_filtered.columns:
        df_filtered = df_filtered[df_filtered['Month'].str.lower() == filters['month'].lower()]
    if 'year' in filters and 'Year' in df_filtered.columns:
        df_filtered = df_filtered[df_filtered['Year'] == filters['year']]

    # Extract entities to compare (cities, departments, doctors, etc.)
    entities = []
    all_cities = set(df['Cities'].dropna().str.lower().unique()) if 'Cities' in df.columns else set()
    all_depts = set(df['Admitting Department'].dropna().str.lower().unique()) if 'Admitting Department' in df.columns else set()
    all_doctors = set(df['Doctor Name'].dropna().str.lower().unique()) if 'Doctor Name' in df.columns else set()
    all_groups = set(df['Group'].dropna().str.lower().unique()) if 'Group' in df.columns else set()

    # Find matching entities in the query
    for city in all_cities:
        if city in q_lower or city.replace(' ', '') in q_lower:
            entities.append(('Cities', city))

    for dept in all_depts:
        if dept in q_lower or dept.replace(' ', '') in q_lower:
            entities.append(('Admitting Department', dept))

    for doctor in all_doctors:
        if doctor in q_lower or doctor.replace(' ', '') in q_lower:
            entities.append(('Doctor Name', doctor))

    for group in all_groups:
        if group in q_lower or group.replace(' ', '') in q_lower:
            entities.append(('Group', group))

    if len(entities) >= 2:
        # Compare the first two entities found
        col1, val1 = entities[0]
        col2, val2 = entities[1]

        rev1 = df_filtered[df_filtered[col1].str.lower() == val1]['Revenue'].sum()
        rev2 = df_filtered[df_filtered[col2].str.lower() == val2]['Revenue'].sum()

        comparison_text = f"Revenue Comparison:\n\n"
        comparison_text += f"{val1.title()}: ₹{rev1/100000:,.2f} lakhs\n"
        comparison_text += f"{val2.title()}: ₹{rev2/100000:,.2f} lakhs\n\n"
        comparison_text += f"Difference: ₹{abs(rev1-rev2)/100000:,.2f} lakhs\n"

        if rev1 > rev2:
            comparison_text += f"{val1.title()} has ₹{(rev1-rev2)/100000:,.2f} lakhs more revenue"
        elif rev2 > rev1:
            comparison_text += f"{val2.title()} has ₹{(rev2-rev1)/100000:,.2f} lakhs more revenue"
        else:
            comparison_text += "Both have equal revenue"

        return [], None, comparison_text

    return [], None, "Could not identify entities to compare. Please specify two items to compare (e.g., 'compare chennai and mumbai revenue')."


def handle_top_n_query(question, filters):
    """Handle top N queries like 'top 5 doctors by revenue' or just 'top doctor'"""
    global df
    q_lower = question.lower()

    # Extract the number (N) from the query, default to 1 if not specified
    import re
    n_match = re.search(r'top\s+(\d+)', q_lower)
    if n_match:
        n = int(n_match.group(1))
    else:
        # If no number specified, assume they want the top 1
        n = 1

    # Apply date filters first
    df_filtered = df.copy()
    if 'date' in filters and 'Date' in df_filtered.columns:
        df_filtered = df_filtered[df_filtered['Date'].dt.date == filters['date']]
    if 'month' in filters and 'Month' in df_filtered.columns:
        df_filtered = df_filtered[df_filtered['Month'].str.lower() == filters['month'].lower()]
    if 'year' in filters and 'Year' in df_filtered.columns:
        df_filtered = df_filtered[df_filtered['Year'] == filters['year']]

    # Determine what to rank by
    grouping_column = None
    if "doctor" in q_lower:
        grouping_column = "Doctor Name"
    elif "department" in q_lower or "dept" in q_lower:
        grouping_column = "Admitting Department"
    elif "city" in q_lower or "cities" in q_lower:
        grouping_column = "Cities"
    elif "service" in q_lower:
        grouping_column = "Service"
    elif "group" in q_lower:
        grouping_column = "Group"
    elif "header" in q_lower:
        grouping_column = "Header"

    if grouping_column and grouping_column in df_filtered.columns:
        # Get top N by revenue
        top_n = (
            df_filtered.groupby(grouping_column)['Revenue']
            .sum()
            .reset_index()
            .sort_values('Revenue', ascending=False)
            .head(n)
        )

        top_n['Revenue(₹ lakhs)'] = top_n['Revenue'] / 100000

        if n == 1:
            # Special handling for single top result
            top_entity = top_n.iloc[0]
            summary_text = f"The top {grouping_column.lower()} by revenue is {top_entity[grouping_column]} with ₹{top_entity['Revenue(₹ lakhs)']:.2f} lakhs."
        else:
            summary_text = f"Top {n} {grouping_column.lower()}s by Revenue:\n\n"
            for i, (_, row) in enumerate(top_n.iterrows(), 1):
                summary_text += f"{i}. {row[grouping_column]}: ₹{row['Revenue(₹ lakhs)']:.2f} lakhs\n"

        rows = top_n[[grouping_column, 'Revenue(₹ lakhs)']].to_dict('records')
        return rows, grouping_column, summary_text

    return [], None, f"Could not determine what to rank. Please specify (e.g., 'top 5 doctors by revenue')."


def handle_which_query(question, filters):
    """Handle 'which' questions like 'Which city had the highest revenue?'"""
    global df
    q_lower = question.lower()

    # Apply date filters first
    df_filtered = df.copy()
    if 'date' in filters and 'Date' in df_filtered.columns:
        df_filtered = df_filtered[df_filtered['Date'].dt.date == filters['date']]
    if 'month' in filters and 'Month' in df_filtered.columns:
        df_filtered = df_filtered[df_filtered['Month'].str.lower() == filters['month'].lower()]
    if 'year' in filters and 'Year' in df_filtered.columns:
        df_filtered = df_filtered[df_filtered['Year'] == filters['year']]
    if 'quarter' in filters and 'Qtr' in df_filtered.columns:
        df_filtered = df_filtered[df_filtered['Qtr'] == filters['quarter']]

    # Extract specific filters from the query (cities, departments, doctors, etc.)
    # Check for city names in the query
    all_cities = set(df['Cities'].dropna().str.lower().unique()) if 'Cities' in df.columns else set()
    for city in all_cities:
        if city in q_lower or city.replace(' ', '') in q_lower:
            df_filtered = df_filtered[df_filtered['Cities'].str.lower() == city]
            break

    # Check for department names
    all_depts = set(df['Admitting Department'].dropna().str.lower().unique()) if 'Admitting Department' in df.columns else set()
    for dept in all_depts:
        if dept in q_lower or dept.replace(' ', '') in q_lower:
            df_filtered = df_filtered[df_filtered['Admitting Department'].str.lower() == dept]
            break

    # Check for doctor names
    all_doctors = set(df['Doctor Name'].dropna().str.lower().unique()) if 'Doctor Name' in df.columns else set()
    for doctor in all_doctors:
        if doctor in q_lower or doctor.replace(' ', '') in q_lower:
            df_filtered = df_filtered[df_filtered['Doctor Name'].str.lower() == doctor]
            break

    # Determine what to find (highest/lowest) and for which entity
    is_lowest = any(word in q_lower for word in ["lowest", "minimum", "min", "least","High","low"])

    if "city" in q_lower or "cities" in q_lower:
        grouping_column = "Cities"
    elif "department" in q_lower or "dept" in q_lower:
        grouping_column = "Admitting Department"
    elif "doctor" in q_lower:
        grouping_column = "Doctor Name"
    elif "service" in q_lower:
        grouping_column = "Service"
    elif "group" in q_lower:
        grouping_column = "Group"
    elif "header" in q_lower:
        grouping_column = "Header"
    else:
        # Default to finding the highest/lowest revenue entity overall
        if "revenue" in q_lower:
            # Find the entity with highest/lowest revenue
            all_entities = []
            for col in ['Cities', 'Admitting Department', 'Doctor Name', 'Service', 'Group', 'Header']:
                if col in df_filtered.columns:
                    entity_revenue = df_filtered.groupby(col)['Revenue'].sum().reset_index()
                    entity_revenue['entity_type'] = col
                    all_entities.append(entity_revenue)

            if all_entities:
                combined = pd.concat(all_entities)
                if is_lowest:
                    extreme_entity = combined.loc[combined['Revenue'].idxmin()]
                    desc = "lowest"
                else:
                    extreme_entity = combined.loc[combined['Revenue'].idxmax()]
                    desc = "highest"

                revenue_lakhs = extreme_entity['Revenue'] / 100000

                summary_text = f"The {extreme_entity['entity_type'].lower()} with the {desc} revenue is {extreme_entity[extreme_entity['entity_type']]} with ₹{revenue_lakhs:.2f} lakhs."

                rows = [{
                    extreme_entity['entity_type']: extreme_entity[extreme_entity['entity_type']],
                    'Revenue(₹ lakhs)': revenue_lakhs
                }]
                return rows, extreme_entity['entity_type'], summary_text

        return [], None, "Could not determine what to find the maximum for. Please specify (e.g., 'which city had the highest revenue?')."

    if grouping_column and grouping_column in df_filtered.columns:
        # Find the entity with maximum/minimum revenue
        if is_lowest:
            extreme_entity = (
                df_filtered.groupby(grouping_column)['Revenue']
                .sum()
                .reset_index()
                .sort_values('Revenue', ascending=True)  # ascending for lowest
                .head(1)
            )
            desc = "lowest"
        else:
            extreme_entity = (
                df_filtered.groupby(grouping_column)['Revenue']
                .sum()
                .reset_index()
                .sort_values('Revenue', ascending=False)  # descending for highest
                .head(1)
            )
            desc = "highest"

        if not extreme_entity.empty:
            revenue_lakhs = extreme_entity['Revenue'].iloc[0] / 100000
            entity_name = extreme_entity[grouping_column].iloc[0]

            # Build summary text with filters
            summary_text = f"The {grouping_column.lower()} with the {desc} revenue"

            if 'month' in filters:
                summary_text += f" in {filters['month']}"
            if 'year' in filters:
                summary_text += f" {filters['year']}"
            if 'quarter' in filters:
                summary_text += f" in Q{filters['quarter']}"

            summary_text += f" is {entity_name} with ₹{revenue_lakhs:.2f} lakhs."

            rows = [{
                grouping_column: entity_name,
                'Revenue(₹ lakhs)': revenue_lakhs
            }]
            return rows, grouping_column, summary_text

    return [], None, f"Could not find data for the specified query."


def handle_who_query(question, filters):
    """Handle 'who' questions like 'who is the top doctor in Q1?'"""
    global df
    q_lower = question.lower()

    # Apply date filters first
    df_filtered = df.copy()
    if 'date' in filters and 'Date' in df_filtered.columns:
        df_filtered = df_filtered[df_filtered['Date'].dt.date == filters['date']]
    if 'month' in filters and 'Month' in df_filtered.columns:
        df_filtered = df_filtered[df_filtered['Month'].str.lower() == filters['month'].lower()]
    if 'year' in filters and 'Year' in df_filtered.columns:
        df_filtered = df_filtered[df_filtered['Year'] == filters['year']]
    if 'quarter' in filters and 'Qtr' in df_filtered.columns:
        df_filtered = df_filtered[df_filtered['Qtr'] == filters['quarter']]

    # Determine what to find the top for
    grouping_column = None
    if "doctor" in q_lower:
        grouping_column = "Doctor Name"
    elif "department" in q_lower or "dept" in q_lower:
        grouping_column = "Admitting Department"
    elif "city" in q_lower or "cities" in q_lower:
        grouping_column = "Cities"
    elif "service" in q_lower:
        grouping_column = "Service"
    elif "group" in q_lower:
        grouping_column = "Group"
    elif "header" in q_lower:
        grouping_column = "Header"

    if grouping_column and grouping_column in df_filtered.columns:
        # Find the top entity by revenue
        top_entity = (
            df_filtered.groupby(grouping_column)['Revenue']
            .sum()
            .reset_index()
            .sort_values('Revenue', ascending=False)
            .head(1)
        )

        if not top_entity.empty:
            revenue_lakhs = top_entity['Revenue'].iloc[0] / 100000
            entity_name = top_entity[grouping_column].iloc[0]

            # Build response text
            response_text = f"The top {grouping_column.lower()}"

            if 'month' in filters:
                response_text += f" in {filters['month']}"
            if 'year' in filters:
                response_text += f" {filters['year']}"
            if 'quarter' in filters:
                response_text += f" in Q{filters['quarter']}"

            response_text += f" is {entity_name} with ₹{revenue_lakhs:.2f} lakhs in revenue."

            rows = [{
                grouping_column: entity_name,
                'Revenue(₹ lakhs)': revenue_lakhs
            }]
            return rows, grouping_column, response_text

    return [], None, f"Could not determine what to find for the 'who' question."


def handle_list_query(question, filters):
    """Handle 'list' questions like 'List doctors under operation theatre'"""
    global df
    q_lower = question.lower()

    # Apply date filters first
    df_filtered = df.copy()
    if 'date' in filters and 'Date' in df_filtered.columns:
        df_filtered = df_filtered[df_filtered['Date'].dt.date == filters['date']]
    if 'month' in filters and 'Month' in df_filtered.columns:
        df_filtered = df_filtered[df_filtered['Month'].str.lower() == filters['month'].lower()]
    if 'year' in filters and 'Year' in df_filtered.columns:
        df_filtered = df_filtered[df_filtered['Year'] == filters['year']]
    if 'quarter' in filters and 'Qtr' in df_filtered.columns:
        df_filtered = df_filtered[df_filtered['Qtr'] == filters['quarter']]

    # Extract what to list and the filter criteria
    list_column = None
    filter_column = None
    filter_value = None

    if "doctor" in q_lower or "doctors" in q_lower:
        list_column = "Doctor Name"

        # Check for department filters
        for dept in df['Admitting Department'].dropna().unique():
            if dept.lower() in q_lower or dept.lower().replace(' ', '') in q_lower:
                filter_column = "Admitting Department"
                filter_value = dept
                break

        # Check for group/header filters
        if not filter_value:
            for group in df['Group'].dropna().unique():
                if group.lower() in q_lower or group.lower().replace(' ', '') in q_lower:
                    filter_column = "Group"
                    filter_value = group
                    break

        if not filter_value:
            for header in df['Header'].dropna().unique():
                if header.lower() in q_lower or header.lower().replace(' ', '') in q_lower:
                    filter_column = "Header"
                    filter_value = header
                    break

    if list_column and list_column in df_filtered.columns:
        if filter_column and filter_value and filter_column in df_filtered.columns:
            # Filter by the specified criteria
            df_filtered = df_filtered[df_filtered[filter_column].str.lower() == filter_value.lower()]

        # Get unique values
        unique_items = df_filtered[list_column].dropna().unique().tolist()
        unique_items.sort()

        if unique_items:
            response_text = f"List of {list_column.lower()}s"
            if filter_column and filter_value:
                response_text += f" under {filter_value}"

            if 'month' in filters:
                response_text += f" in {filters['month']}"
            if 'year' in filters:
                response_text += f" {filters['year']}"

            response_text += f": {', '.join(unique_items)}"

            rows = [{list_column: item} for item in unique_items]
            return rows, list_column, response_text

    return [], None, f"Could not find data for the list query."


def process_query_aggregation(question):
    global df

    filters = extract_date_filters(question)
    q_lower = question.lower()
    grouping_column = None

    # Check for comparison queries
    if any(word in q_lower for word in ["compare", "comparison", "vs", "versus", "difference", "more than", "less than", "higher", "lower"]):
        return handle_comparison_query(question, filters)

    # Check for top N queries
    if "top" in q_lower:
        return handle_top_n_query(question, filters)

    # Check for "which" questions (highest, most, etc.)
    if any(word in q_lower for word in ["which", "what", "highest", "most", "maximum", "max", "best", "lowest", "minimum", "min", "least"]):
        return handle_which_query(question, filters)

    # Check for "who" questions (who is the top doctor, etc.)
    if "who" in q_lower and any(word in q_lower for word in ["top", "best", "highest", "most"]):
        return handle_who_query(question, filters)

    # Check for "list" questions (list doctors under operation theatre, etc.)
    if "list" in q_lower and any(word in q_lower for word in ["doctor", "doctors", "under", "in", "at"]):
        return handle_list_query(question, filters)

    if "doctor" in q_lower:
        grouping_column = "Doctor Name"
    elif ("department" in q_lower or "dept" in q_lower) and "wise" in q_lower:
        grouping_column = "Admitting Department"
    elif ("department" in q_lower or "dept" in q_lower) and ("revenue" in q_lower or "rev" in q_lower or "wise" in q_lower):
        grouping_column = "Admitting Department"
    elif "department" in q_lower or "dept" in q_lower:
        grouping_column = "Admitting Department"
    elif "city" in q_lower or "cities" in q_lower:
        grouping_column = "Cities"
    elif "service" in q_lower:
        grouping_column = "Service"
    elif "group" in q_lower:
        grouping_column = "Group"
    elif "header" in q_lower:
        grouping_column = "Header"
    elif "day" in q_lower or "daily" in q_lower:
        grouping_column = "Day"
    elif "month" in q_lower or "monthly" in q_lower:
        grouping_column = "Month"
    elif "quarter" in q_lower or "qtr" in q_lower:
        grouping_column = "Qtr"
    elif "year" in q_lower or "yearly" in q_lower:
        grouping_column = "Year"

    df_filtered = df.copy()
    filtered_columns = set()  # Track which columns we've filtered by

    # Apply filters step by step
    if 'date' in filters and 'Date' in df_filtered.columns:
        df_filtered = df_filtered[df_filtered['Date'].dt.date == filters['date']]
        filtered_columns.add('Date')

    if 'month' in filters and 'Month' in df_filtered.columns:
        df_filtered = df_filtered[df_filtered['Month'].str.lower() == filters['month'].lower()]
        filtered_columns.add('Month')

    if 'year' in filters and 'Year' in df_filtered.columns:
        df_filtered = df_filtered[df_filtered['Year'] == filters['year']]
        filtered_columns.add('Year')

    if 'quarter' in filters and 'Qtr' in df_filtered.columns:
        df_filtered = df_filtered[df_filtered['Qtr'] == filters['quarter']]
        filtered_columns.add('Quarter')

    if 'day' in filters and 'Day' in df_filtered.columns:
        df_filtered = df_filtered[df_filtered['Day'] == filters['day']]
        filtered_columns.add('Day')

    if 'start_time' in filters and 'Hour' in df_filtered.columns:
        start_hr = filters['start_time'].hour
        end_hr = filters['end_time'].hour if 'end_time' in filters else start_hr
        df_filtered = df_filtered[(df_filtered['Hour'] >= start_hr) & (df_filtered['Hour'] <= end_hr)]
        filtered_columns.add('Hour')

    # Extract and apply specific value filters (cities, doctors, departments, etc.)
    # Check for city names in the query
    all_cities = set(df['Cities'].dropna().str.lower().unique()) if 'Cities' in df.columns else set()
    for city in all_cities:
        if city in q_lower or city.replace(' ', '') in q_lower:
            df_filtered = df_filtered[df_filtered['Cities'].str.lower() == city]
            print(f"Filtered by city: {city}")
            break

    # Check for department names
    all_depts = set(df['Admitting Department'].dropna().str.lower().unique()) if 'Admitting Department' in df.columns else set()
    for dept in all_depts:
        if dept in q_lower or dept.replace(' ', '') in q_lower:
            df_filtered = df_filtered[df_filtered['Admitting Department'].str.lower() == dept]
            print(f"Filtered by department: {dept}")
            break

    # Check for doctor names
    all_doctors = set(df['Doctor Name'].dropna().str.lower().unique()) if 'Doctor Name' in df.columns else set()
    for doctor in all_doctors:
        if doctor in q_lower or doctor.replace(' ', '') in q_lower:
            df_filtered = df_filtered[df_filtered['Doctor Name'].str.lower() == doctor]
            print(f"Filtered by doctor: {doctor}")
            filtered_by_specific_value = True
            break

    # Check for group names
    all_groups = set(df['Group'].dropna().str.lower().unique()) if 'Group' in df.columns else set()
    for group in all_groups:
        group_no_spaces = group.replace(' ', '')
        if (group in q_lower or
            group_no_spaces in q_lower or
            any(word.lower() in group for word in q_lower.split()) or
            any(word.lower() in group_no_spaces for word in q_lower.split())):
            df_filtered = df_filtered[df_filtered['Group'].str.lower() == group]
            filtered_by_specific_value = True
            break

    # Check for header names
    all_headers = set(df['Header'].dropna().str.lower().unique()) if 'Header' in df.columns else set()
    for header in all_headers:
        if header in q_lower or header.replace(' ', '') in q_lower:
            df_filtered = df_filtered[df_filtered['Header'].str.lower() == header]
            print(f"Filtered by header: {header}")
            filtered_by_specific_value = True
            break

    # Check for service names
    all_services = set(df['Service'].dropna().str.lower().unique()) if 'Service' in df.columns else set()
    for service in all_services:
        if service in q_lower or service.replace(' ', '') in q_lower:
            df_filtered = df_filtered[df_filtered['Service'].str.lower() == service]
            print(f"Filtered by service: {service}")
            filtered_by_specific_value = True
            break

    # If no rows found
    if df_filtered.empty:
        return [], None, "No data found for the given query or time period."

    # Compute total revenue
    if 'Total Revenue' in df_filtered.columns:
        total_revenue = df_filtered['Total Revenue'].sum()
    elif 'Revenue' in df_filtered.columns:
        total_revenue = df_filtered['Revenue'].sum()
    else:
        return "Revenue column not found in data."

    # Format summary text
    if grouping_column == "Admitting Department" and ("revenue" in q_lower or "rev" in q_lower or "wise" in q_lower):
        summary_text = "Revenue by Department"
    else:
        summary_text = "Total Revenue"

    if 'date' in filters:
        summary_text += f" on {filters['date'].strftime('%B %d, %Y')}"
    elif 'months' in filters and 'year' in filters:
        if len(filters['months']) == 2:
            months_str = f"{filters['months'][0]} and {filters['months'][1]}"
        else:
            months_str = ", ".join(filters['months'][:-1]) + f" and {filters['months'][-1]}"
        summary_text += f" in {months_str} {filters['year']}"
    elif 'months' in filters:
        if len(filters['months']) == 2:
            months_str = f"{filters['months'][0]} and {filters['months'][1]}"
        else:
            months_str = ", ".join(filters['months'][:-1]) + f" and {filters['months'][-1]}"
        summary_text += f" in {months_str}"
    elif 'month' in filters and 'year' in filters:
        summary_text += f" in {filters['month']} {filters['year']}"
    elif 'month' in filters:
        summary_text += f" in {filters['month']}"
    elif 'quarter' in filters and 'year' in filters:
        summary_text += f" in Q{filters['quarter']} {filters['year']}"
    elif 'year' in filters:
        summary_text += f" in {filters['year']}"

    if 'start_time' in filters:
        summary_text += f" between {filters['start_time'].strftime('%I:%M %p')}"
        if 'end_time' in filters:
            summary_text += f" and {filters['end_time'].strftime('%I:%M %p')}"

    if not (grouping_column == "Admitting Department" and ("revenue" in q_lower or "rev" in q_lower or "wise" in q_lower)):
        summary_text += f": ₹{total_revenue/100000:,.2f} lakhs"

    rows = []
    # --- Grouped breakdown ---
    # Only create breakdown if we haven't already filtered by this column
    if (grouping_column and
        grouping_column in df_filtered.columns and
        grouping_column not in filtered_columns):
        df_grouped = (
            df_filtered.groupby(grouping_column)['Revenue']
            .sum()
            .reset_index()
            .sort_values('Revenue', ascending=False)
        )
        df_grouped['Revenue(₹ lakhs)'] = df_grouped['Revenue'] / 100000
        rows = df_grouped[[grouping_column, 'Revenue(₹ lakhs)']].to_dict('records')

        # If asking for revenue by department, show all departments
        if grouping_column == "Admitting Department" and ("revenue" in q_lower or "rev" in q_lower or "wise" in q_lower):
            # Don't limit to just the top ones, show all
            pass

        # Optional: append breakdown to summary_text (removed to keep only simple answer)
        # if rows:
        #     summary_text += f"\n\n**Total Revenue by {grouping_column}**\n\n| {grouping_column} | Revenue(₹ lakhs) |\n|{'-'*len(grouping_column)}|{'-'*18}|"
        #     for r in rows:
        #         summary_text += f"\n| {r[grouping_column]} | {r['Revenue(₹ lakhs)']:,.2f} |"

    # Optional: Add department-level breakdown (removed as per user request)
    # if 'Department' in df_filtered.columns:
    #     department_summary = (
    #         df_filtered.groupby('Department')['Total Revenue']
    #         .sum()
    #         .reset_index()
    #         .sort_values(by='Total Revenue', ascending=False)
    #     )

    #     dept_table = "\n\n**Revenue by Department (₹ lakhs)**\n\n| Department | Total Revenue |\n|-------------|----------------:|"
    #     for _, row in department_summary.iterrows():
    #         dept_table += f"\n| {row['Department']} | {row['Total Revenue']/100000:,.2f} |"

    #     summary_text += dept_table

    return rows, grouping_column, summary_text

def initialize_app():
    """Initialize all components of the application."""
    print("Initializing application...")
    
    initialize_gemini()
    create_temp_directory()
    load_dataset()
    initialize_embeddings()
    initialize_vectorstore()
    initialize_llm_chain()
    
    print("Application initialized successfully!")


# Routes
@app.route('/')
def index():
    return send_file('index.html')


@app.route('/ask', methods=['POST'])
def ask():
    data = request.get_json()
    question = data.get('question', '').strip()

    if not question:
        return jsonify({'answer': 'Please provide a question.'})

    # Normalize and domain check
    question_norm = normalize_query(question, ALIASES)
    if not is_in_domain(question_norm):
        return jsonify({'answer': 'Sorry, I can only answer questions related to billing data.'})

    # Enhanced aggregation logic
    rows, grouping_column, summary_text = process_query_aggregation(question_norm)
    distinct_values = None  # If needed

    if summary_text:
        # If aggregation provided an answer (e.g., for date/revenue queries), use it directly
        final_answer = summary_text
    else:
        # Otherwise, use LLM
        result = qa_chain({"query": question_norm})
        answer = result["result"]
        final_answer = answer

    # Generate dynamic suggestions based on the current query result
    dynamic_suggestions = []
    if rows and grouping_column:
        # Get top 3 items from the results for suggestions
        top_items = rows[:3] if len(rows) >= 3 else rows
        for item in top_items:
            if grouping_column in item:
                entity_name = item[grouping_column]
                if grouping_column == 'Admitting Department':
                    dynamic_suggestions.append(f"revenue of {entity_name.lower()}")
                elif grouping_column == 'Doctor Name':
                    dynamic_suggestions.append(f"revenue of {entity_name.lower()}")
                elif grouping_column == 'Cities':
                    dynamic_suggestions.append(f"revenue of {entity_name.lower()}")

        # Add advanced suggestions based on the grouping column
        if grouping_column == 'Cities':
            dynamic_suggestions.extend([
                "top 5 cities by revenue",
                "compare revenue with last month",
                "show city with lowest revenue"
            ])
        elif grouping_column == 'Admitting Department':
            dynamic_suggestions.extend([
                "top 5 departments by revenue",
                "compare department revenue with last quarter",
                "show department with lowest revenue"
            ])
        elif grouping_column == 'Doctor Name':
            dynamic_suggestions.extend([
                "top 5 doctors by revenue",
                "compare doctor revenue with last month",
                "show doctor with lowest revenue"
            ])

    return jsonify({
        'answer': final_answer,
        'rows': rows,
        'distinct_values': distinct_values,
        'grouping_column': grouping_column,
        'dynamic_suggestions': dynamic_suggestions
    })


@app.route('/api/favourites', methods=['GET', 'POST'])
def handle_favourites():
    if request.method == 'POST':
        data = request.get_json()
        question = data.get('question', '').strip()
        if question and question not in favourites:
            favourites.append(question)
        return jsonify({'status': 'ok'})
    else:
        return jsonify({'favourites': favourites})




@app.route('/api/distinct_values', methods=['GET'])
def distinct_values():
    """Returns distinct values for a specified column, supporting short forms."""
    column_param = request.args.get('column', '').strip().lower()

    if not column_param:
        return jsonify({'error': 'Column parameter is required'}), 400

    column_name = COLUMN_ALIASES.get(column_param, column_param.title())

    if column_name not in df.columns:
        return jsonify({'error': f'Column "{column_name}" not found in dataset'}), 400

    distinct_vals = df[column_name].dropna().unique().tolist()
    distinct_vals.sort()

    return jsonify({
        'column': column_name,
        'distinct_values': distinct_vals,
        'count': len(distinct_vals)
    })


@app.route('/suggestions', methods=['GET'])
def get_suggestions():
    """Returns auto-suggestions based on partial input."""
    partial = request.args.get('q', '').strip().lower()
    if not partial:
        return jsonify({'suggestions': []})

    suggestions = set()

    # Check aliases
    for alias, full in ALIASES.items():
        if alias.lower().startswith(partial):
            suggestions.add(full)

    # Check distinct values in dataset
    for col, values in DISTINCT_VALUES.items():
        for val in values:
            if isinstance(val, str) and val.lower().startswith(partial):
                suggestions.add(val)

    # Also check column names and aliases
    for alias, full in COLUMN_ALIASES.items():
        if alias.lower().startswith(partial):
            suggestions.add(full)

    # Limit to top 10 suggestions
    suggestions = list(suggestions)[:10]

    return jsonify({'suggestions': suggestions})

@app.route('/transcribe', methods=['POST'])
def transcribe_audio():
    """Receives audio data, sends it to Gemini, and returns the transcription."""
    print("\n--- New Transcription Request Received ---")
    
    # Handle FormData from index.html
    if 'audio' in request.files:
        audio_file = request.files['audio']
        audio_bytes = audio_file.read()
        language = request.form.get('languageCode', 'en-US')
    else:
        # Fallback for JSON
        data = request.get_json()
        if not data or 'audio' not in data:
            print("Error: No audio data in request.")
            return jsonify({'text': '', 'error': 'No audio data provided'}), 400
        
        audio_base64 = data['audio']
        language = data.get('languageCode', 'en-US')
        
        try:
            audio_bytes = base64.b64decode(audio_base64)
        except Exception as e:
            print(f"Error decoding base64 audio: {e}")
            return jsonify({'text': '', 'error': f'Failed to process audio file: {e}'}), 500
    
    print(f"Received language code from frontend: {language}")
    
    try:
        if not os.getenv("GEMINI_API_KEY"):
            print("Error: GEMINI_API_KEY environment variable is not set.")
            return jsonify({'text': '', 'error': 'GEMINI_API_KEY is not set'}), 500
        
        audio_part = {"mime_type": "audio/webm", "data": audio_bytes}
        model = genai.GenerativeModel(model_name='models/gemini-2.5-flash')
        
        prompt = f"""You are an expert ASR and query normalizer for a hospital revenue dataset.

        The audio is in {language} and may contain:
        - Medical terminology (departments, services, doctor names)
        - City names and abbreviations (chn=Chennai, cbe=Coimbatore, bng=Bengaluru)  
        - Healthcare abbreviations (cardio=Cardiology, ortho=Orthopedics, gastro=Gastroenterology)
        - Revenue/billing terms (rev=Revenue, dept=Department)
        - Misspellings and informal speech patterns

        Instructions:
        1. Transcribe the audio accurately, handling medical and regional terminology
        2. Normalize abbreviations and correct common misspellings:
        - City abbreviations: chn→Chennai, cbe→Coimbatore, bng→Bengaluru
        - Department abbreviations: cardio→Cardiology, ortho→Orthopedics, gastro→Gastroenterology  
        - Doctor name patterns: recognize partial names and expand properly
        - Service abbreviations: xray→X-Ray, lab→Laboratory, ultra→Ultrasound
        3. Translate to fluent American English if needed
        4. Add "revenue" context if the query is about financial/billing data but doesn't explicitly mention it
        5. Return ONLY the final normalized English query - no explanations or metadata

        Expected output: Clean, properly spelled, healthcare-appropriate English query ready for database search."""
        
        print("Sending request to Gemini API...")
        response = model.generate_content([prompt, audio_part])
        
        try:
            response.resolve()
        except Exception as e:
            print(f"Note: response.resolve() failed, but this might be okay. Error: {e}")
            pass
        
        transcription = (getattr(response, 'text', '') or '').strip()
        print(f"Received transcription from Gemini: '{transcription}'")
        
        if not transcription:
            print("Warning: Empty transcription received from model.")
            print(f"Full Gemini Response Parts: {response.parts}")
            print(f"Prompt Feedback: {response.prompt_feedback}")
            return jsonify({'text': '', 'error': 'Empty transcription received. The audio might be silent or the content was blocked.'}), 502
        
        return jsonify({'text': transcription})
    
    except Exception as e:
        print(f"CRITICAL Error during transcription with Gemini: {e}")
        return jsonify({'text': '', 'error': f'Gemini API error: {str(e)}'}), 500

if __name__ == "__main__":
    initialize_app()
    # Use threaded=False on Windows to avoid WinError 10038
    app.run(debug=True, port=5000, threaded=False)


