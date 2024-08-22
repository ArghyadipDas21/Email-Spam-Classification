import pickle
import streamlit as st
model=pickle.load(open("spam.pkl","rb"))
cv=pickle.load(open("vectorizer.pkl","rb"))

def main():
    st.title("Arghyadip's Email Spam classification App")
    st.subheader("Build by Arghyadip Das with Python and StreamLit")
    msg= st.text_input("Enter a text")
    if st.button("Predict"):
        data=[msg]
        vect= cv.transform(data).toarray()
        prediction=model.predict(vect)
        result=prediction[0]
        if result==1:
            st.error("this is a spam mail")
        else:
            st.success("this is Not a spam mail, you can click on it")


main()
