import pandas as pd
from openai.embeddings_utils import cosine_similarity

# Load the dataset (assuming it's a CSV)
df = pd.read_csv("Heart_Lung_and_BloodQA.csv")

# Preprocess the data (remove punctuation, lowercase, etc.)
df['Question'] = df['Question'].astype(str).str.lower().str.replace('[^\w\s]', '', regex=True)

# Initialize OpenAI API key

# Function to get embedding
def get_embedding(text, model="text-embedding-ada-002"):
   return openai.Embedding.create(input = [text], model=model)['data'][0]['embedding']

# Pre-calculate embeddings for all questions in the dataset
df['Question_Embedding'] = df['Question'].apply(get_embedding)

# Save the DataFrame with embeddings and similarities to a CSV file
df.to_csv("qa_dataset_with_embeddings.csv", index=False)  # index=False to avoid saving row numbers
import ast
df = pd.read_csv("qa_dataset_with_embeddings.csv")

# Convert the string embeddings back to lists
df['Question_Embedding'] = df['Question_Embedding'].apply(ast.literal_eval)

def find_best_answer(user_question):
   # Get embedding for the user's question
   user_question_embedding = get_embedding(user_question)

   # Calculate cosine similarities for all questions in the dataset
   df['Similarity'] = df['Question_Embedding'].apply(lambda x: cosine_similarity(x, user_question_embedding))

   # Find the most similar question and get its corresponding answer
   most_similar_index = df['Similarity'].idxmax()
   max_similarity = df['Similarity'].max()

   # Set a similarity threshold to determine if a question is relevant enough
   similarity_threshold = 0.6  # You can adjust this value

   if max_similarity >= similarity_threshold:
      best_answer = df.loc[most_similar_index, 'Answer']
      return best_answer
   else:
      return "I apologize, but I don't have information on that topic yet. Could you please ask other questions?"


# Streamlit app
st.title("Question Answering App")

user_question = st.text_input("Ask your question:")
if st.button("Search"):
    if user_question:
        answer = find_best_answer(user_question)
        st.write(answer)
