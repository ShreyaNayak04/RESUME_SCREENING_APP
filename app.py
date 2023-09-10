import nltk
import re
import pickle
import streamlit as st 
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(stop_words='english')

nltk.download('punkt')
nltk.download('stopwords')

#loading moels
clf = pickle.load(open('clf.pkl','rb'))
tfidf = pickle.load(open('tfidf.pkl','rb'))

def cleanResume(txt):
    cleanTxt = re.sub('http\S+\s',' ',txt)
    cleanTxt = re.sub('RT|CC',' ',cleanTxt)
    cleanTxt = re.sub('@\S+',' ',cleanTxt)
    cleanTxt = re.sub('#\S+',' ',cleanTxt)
    cleanTxt = re.sub('[%s]' % re.escape("""|"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""),' ',cleanTxt)
    cleanTxt = re.sub(r'[^\x000-\x7f]',' ',cleanTxt)
    cleanTxt = re.sub('\s+',' ',cleanTxt)
    return cleanTxt


#App
def main():
    st.title("Resume Screening App") #Title of the app
    st.write("Please upload your resume here : ")
    upload_file = st.file_uploader('Upload Resume', type=['txt','pdf'])
    
    if upload_file is not None:
        try:
            resume_bytes = upload_file.read()
            resume_txt = resume_bytes.decode('utf-8')
        except UnicodeDecodeError:
            resume_txt = resume_bytes.decode('latin-1')    
        
        cleaned_reusme = cleanResume(resume_txt)    
        final_cleaned_resume = tfidf.transform([cleaned_reusme])
        prediction_id = clf.predict(final_cleaned_resume)[0]
        st.write(prediction_id)
        
        
        category_mapping = {
        15: "Java Developer",
        23: "Testing",
        8: "DevOps Engineer",
        20: "Python Developer",
        24: "Web Designing",
        12: "HR",
        13: "Hadoop",
        3: "Blockchain",
        10: "ETL Developer",
        18: "Operations Manager",
        6: "Data Science",
        22: "Sales",
        16: "Mechanical Engineer",
        1: "Arts",
        7: "Database",
        11: "Electrical Engineering",
        14: "Health and fitness",
        19: "PMO",
        4: "Business Analyst",
        9: "DotNet Developer",
        2: "Automation Testing",
        17: "Network Security Engineer",
        21: "SAP Developer",
        5: "Civil Engineer",
        0:  "Advocate",

}

        category_name = category_mapping.get(prediction_id, "Unknown")

        st.write('Predicted Category : ', category_name)
    
    
    
if __name__ == "__main__":
    main()
