import os
import streamlit as st
import pandas as pd
import torch
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DataFrameLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
import difflib

# Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Domain keywords for billing data
BILLING_KEYWORDS = {"billing","revenue","amount","pat count","billtype","regnumber","ipno","patientname",
"dept","admitdoctor","ledger","groupname","headername","orderingdoctor","orderingdept","servicename",
"time","billno","servqty","taxamount","concamt","city","cityid","district",
"state","serviceid","iheader_id","igroup_id","iadmit_doc_id","ord_doc_id","ideptid","ord_dept_id",
"patid","opid","ipid","pricelistid","mrd","userid","pat_type_id","patient type","rate","corperate type",
"corp_type_id","bedno","ward","dynamic amount range","bedtype","financialyear","qtr","quarter","avgrpb",
"discount","docconcamt","net","bed category","standard financial year","date level label","weekday","dr","rev","city names","last year","last month"}

def is_in_domain(text):
    """Checks if the text contains any of the required billing keywords."""
    text_lower = text.lower()
    for keyword in BILLING_KEYWORDS:
        if keyword in text_lower:
            return True
    return False

# Aliases for common abbreviations and typos
ALIASES = {
    # Cities
    "cbe": "coimbatore",
    "kovai":"coimbatore",
    "cbre":"coimbatore",
    "chn": "chennai",
    "chnni": "chennai",
    "coimbtore": "coimbatore",
    "chennia": "chennai",
    "mumbai": "mumbai",
    "delhi": "delhi",
    "bangalore": "bengaluru",
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
    "ortho": "orthopedics",
    "pedia": "pediatrics",
    "gyne": "gynecology",
    "ent": "ent",
    "opthal": "ophthalmology",
    "radio": "radiology",
    "gastro": "gastroenterology",
    "gestro": "gastroenterology",
    "gastrology": "gastroenterology",
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
    "hyderbad": "hyderabad"
}

# Streamlit UI setup
st.set_page_config(page_title="Medical RAG App", layout="wide")
st.title("Bot")

DATA_PATH = r"C:\Users\shobika\New folder\billing data.csv"
CHROMA_DIR = "chroma_db_langchain_v2"

# Load dataset
if not os.path.exists(DATA_PATH):
    st.error("Dataset not found. Please upload medical_dataset.csv.")
    st.stop()

df = pd.read_csv(DATA_PATH)
print(f"Loaded dataset with {len(df)} rows and {len(df.columns)} columns")

# Extract distinct values for all columns for better context
DISTINCT_VALUES = {}
for col in df.columns:
    distinct = df[col].dropna().unique().tolist()
    DISTINCT_VALUES[col] = distinct

# Convert rows into LangChain Documents
loader = DataFrameLoader(df, page_content_column="Admitting Department")  # pick one column as content
docs = loader.load()
print(f"Created {len(docs)} documents from DataFrame")

# Create comprehensive searchable content by including ALL columns for accurate answers
for i, row in df.iterrows():
    # Include ALL columns for complete information and accurate answers
    content_parts = []
    for col in df.columns:
        if pd.notna(row[col]):  # Only include non-null values
            if col == 'Revenue':
                content_parts.append(f"{col}: ₹{row[col]:.2f}")
            else:
                content_parts.append(f"{col}: {row[col]}")

    content = ". ".join(content_parts)
    docs[i].page_content = content
    docs[i].metadata = row.to_dict()

# Split documents with better parameters for medical data
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,  # Larger chunks for better context
    chunk_overlap=200,  # More overlap to maintain context
    separators=[". ", "; ", "\n", " ", ""]  # Better separators for medical data
)
split_docs = text_splitter.split_documents(docs)
print(f"Split into {len(split_docs)} document chunks")

# Use custom embeddings approach to avoid meta device issues
try:
    import torch
    from transformers import AutoTokenizer, AutoModel

    # Load model directly with transformers
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    print(f"Loading {model_name} model...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir="./model_cache")
    model = AutoModel.from_pretrained(model_name, cache_dir="./model_cache")
    print("Model loaded successfully!")

    # Create a simple embedding function
    def get_embeddings(texts):
        inputs = tokenizer(texts, return_tensors='pt', padding=True, truncation=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
        # Move tensor to CPU before converting to numpy
        return outputs.last_hidden_state.mean(dim=1).cpu().numpy()

    # Create a custom embeddings class
    class CustomEmbeddings:
        def embed_documents(self, texts):
            return [get_embeddings([text])[0] for text in texts]

        def embed_query(self, text):
            return get_embeddings([text])[0]

    embeddings = CustomEmbeddings()
    print("Custom embeddings created successfully!")

except Exception as e:
    st.error(f"Could not load embeddings model: {e}")
    st.error("Please check your internet connection and try again.")
    st.stop()


# Chroma vector store
if not os.path.exists(CHROMA_DIR):
    print(f"Creating new vectorstore with {len(split_docs)} documents...")
    vectorstore = Chroma.from_documents(split_docs, embedding=embeddings, persist_directory=CHROMA_DIR)
    print("Vectorstore created and persisted successfully!")
else:
    print("Loading existing vectorstore...")
    vectorstore = Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)
    print(f"Vectorstore loaded with {vectorstore._collection.count()} documents")

retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={
        "k": 15  # Use similarity search for faster retrieval
    }
)

# Groq LLM
if not GROQ_API_KEY:
    st.error("⚠️ Please set GROQ_API_KEY in .env")
    st.stop()

llm = ChatGroq(
    model="gemma2-9b-it",
    groq_api_key=GROQ_API_KEY,
)

# Memory for conversation
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, output_key="answer")

# Conversational RAG Chain with custom prompt for revenue calculations
from langchain.prompts import PromptTemplate

# Custom prompt template for focused, ChatGPT-like responses
template = """You are a specialized medical billing data assistant with knowledge of all distinct values in the dataset. Only answer questions related to billing data.

Context: {context}

Dataset Information:
- Columns: Year, Qtr, Month, Day, Financial Year, Quarter, Admitting Department, Doctor Name, Ordering Doctor, Service, Group, Header, Cities, Revenue
- Distinct values are available for all columns to provide accurate answers

Question: {question}
Chat History: {chat_history}

Instructions:
- Columns in the data: Year, Quarter, Month, Day, Financialyear, Qtr, Admitting Department, Doctor Name, Ordering Doctor, Group, Header, Service, Cities, Revenue
- Respond naturally like ChatGPT.
- Identify if the spelling is wrong, suggest other possible meanings, and recognize partial words.
- For questions about available options, list the relevant distinct values.
- Handle short forms, abbreviations, and spelling mistakes intelligently.
- Calculate revenue totals accurately.
- Format revenue amounts clearly as ₹1,234.56.
- Stay focused on billing data only; do not provide unrelated explanations.
- Be comprehensive, helpful, and concise.
- If the input cannot be matched, suggest the closest possible matches from the dataset.
Answer:"""

QA_PROMPT = PromptTemplate(
    template=template,
    input_variables=["context", "question", "chat_history"]
)

# Conversational RAG Chain
qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    memory=memory,
    return_source_documents=True,
    combine_docs_chain_kwargs={"prompt": QA_PROMPT}
)

# Initialize chat history in session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Display chat history
st.subheader("Chat History")
for q, a in st.session_state.chat_history:
    st.write(f"**Q:** {q}")
    st.write(f"**A:** {a}")
    st.write("---")

# Streamlit input
query = st.text_area("Ask a question about the dataset:", height=100)
if st.button("Ask") and query:
    # Check if query is in domain
    if not is_in_domain(query):
        st.subheader("Answer")
        st.write("Sorry, I can only answer questions related to billing data.")
    else:
        # Use LLM only to interpret intent
        result = qa_chain({"question": query})
        answer = result["answer"]

    st.subheader("Answer")
    st.write(answer)

    # ----------------------------
    # 🔑 Work directly with full DataFrame
    # ----------------------------
    q_lower = query.lower()
    df_result = None
    grouping_column = None

    # Detect grouping column
    if "doctor" in q_lower:
        grouping_column = "Doctor Name" if "Doctor Name" in df.columns else "Ordering Doctor"
    elif "department" in q_lower:
        grouping_column = "Admitting Department"
    elif "city" in q_lower or "cities" in q_lower:
        grouping_column = "Cities"
    elif "service" in q_lower:
        grouping_column = "Service"
    elif "group" in q_lower:
        grouping_column = "Group"
    elif "header" in q_lower:
        grouping_column = "Header"

    # Improved detection: check for values in query using aliases and fuzzy matching for all columns
    if not grouping_column:
        query_words = q_lower.split()
        for word in query_words:
            word_lower = word.lower()
            # Check aliases first
            if word_lower in ALIASES:
                # Determine which column the alias belongs to
                alias_value = ALIASES[word_lower]
                for col in df.columns:
                    if col in ["Cities", "Admitting Department", "Service", "Group", "Header"]:
                        unique_vals = df[col].dropna().unique()
                        if any(alias_value.lower() in val.lower() or val.lower() in alias_value.lower() for val in unique_vals):
                            grouping_column = col
                            break
                if grouping_column:
                    break
            # Then fuzzy match for all relevant columns
            if not grouping_column:
                for col in ["Cities", "Admitting Department", "Service", "Group", "Header"]:
                    if col in df.columns:
                        unique_vals = df[col].dropna().unique()
                        for val in unique_vals:
                            if difflib.SequenceMatcher(None, word_lower, val.lower()).ratio() > 0.7:
                                grouping_column = col
                                break
                        if grouping_column:
                            break
            if grouping_column:
                break

    # If a grouping column is found → aggregate
    if grouping_column and "Revenue" in df.columns:
        # If a specific value is mentioned in the query (like 'Coimbatore')
        unique_vals = df[grouping_column].dropna().unique()
        query_words = q_lower.split()
        filter_values = []
        for val in unique_vals:
            val_lower = val.lower()
            # Exact match first
            if val_lower in q_lower:
                filter_values.append(val)
            else:
                # Check aliases
                for word in query_words:
                    word_lower = word.lower()
                    if word_lower in ALIASES and ALIASES[word_lower].lower() == val_lower:
                        filter_values.append(val)
                        break
                    # Fuzzy match
                    elif difflib.SequenceMatcher(None, word, val_lower).ratio() > 0.7:
                        filter_values.append(val)
                        break

        if filter_values:
            df_result = df[df[grouping_column].str.lower().isin([v.lower() for v in filter_values])]
        else:
            df_result = df.copy()

        if not df_result.empty:
            df_grouped = df_result.groupby(grouping_column)["Revenue"].sum().reset_index()

            st.subheader(f"Revenue by {grouping_column}")
            st.dataframe(df_grouped)

            # Bar chart like Power BI
            st.bar_chart(df_grouped.set_index(grouping_column))
        else:
            st.warning(f"No data found for {grouping_column} filter in query: {query}")
    else:
        st.info("No grouping applied. Showing raw rows instead.")
        st.dataframe(df.head(50))  # Show sample rows