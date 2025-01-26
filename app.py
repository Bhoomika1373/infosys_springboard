import streamlit as st

# Function to handle user queries
def main():
    st.title("Medical Assistant Chatbot")
    st.subheader("Ask a question related to the Healthy Heart document!")

    # User input
    user_query = st.text_input("Enter your query:")

    # Display response
    if st.button("Submit"):
        if user_query.strip():
            with st.spinner("Processing..."):
                try:
                    from backend import response  # Replace 'your_module' with the actual filename of your model script
                    model_response = response(user_query)
                    st.success("Response:")
                    st.write(model_response)
                except Exception as e:
                    st.error(f"An error occurred: {e}")
        else:
            st.warning("Please enter a valid query!")

if __name__ == "__main__":
    main()
