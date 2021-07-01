# -*- coding: utf-8 -*-
import numpy as np
import pickle
import pandas as pd
#from flasgger import Swagger
import streamlit as st 
from PIL import Image
from tensorflow.keras.models import load_model
import joblib
from pathlib import Path
#import SessionState
#app=Flask(__name__)
#Swagger(app)


#@app.route('/')
def welcome():
    return "Welcome All"

artifacts_path = Path.joinpath(Path.cwd(),'model_artifacts')

#@app.route('/predict',methods=["Get"])
def classify_utterance(utt):
    # load the vectorizer
    loaded_vectorizer = joblib.load(Path.joinpath(artifacts_path,'vectorizer.pickle'))

    # load the model
    loaded_model = load_model(Path.joinpath(artifacts_path,'classification.model'))

    # make a prediction
    return(loaded_model.predict(loaded_vectorizer.transform([utt])))



def main():
    st.title("EMAIL TEMPLATE")
    html_temp = """
    <html> 
<head>
<title>EMAIL TEMPLATE</title>
<style>
body {
background-image: url("https://png2.cleanpng.com/sh/1ba3f93f3e3f71c439e5e46c007bdf1d/L0KzQYm3V8EyN5NwiZH0aYP2gLBuTfNwdaF6jNd7LXnmf7B6TfVuaZpxRdN9dHHmeL7sjwQudZJuhJ97dT3vfLS0hvlvbF55Rdk2bXHsfH68gsQ3PWU2fKU6NHPmQ3A5VMEyPGM4S6MAM0G2Q4aAUMkzOGIARuJ3Zx==/kisspng-computer-icons-email-attachment-mail-ru-llc-find-t-g-mail-5b46541d314cc3.2411423315313357092019.png");

background-repeat: no-repeat;
background-size: cover;
}
</style>

    
    <div style="background-color:grey;padding:10px">
    <h2 style="color:white;text-align:center;">EMAIL TEMPLATES SUGGESTION ML App </h2>

    </div>
    
  
            
</html>
    """
    page_bg_img = '''
             <style>
                body {
                    
                       background-image: url("https://www.stkconf.org/wp-content/uploads/2018/10/Web-Page-Background-Color.jpg");
                  background-size: cover;
      
                       }      
                </style>
            '''
    html_java = '''<html>
<head>
<meta http-equiv="Content-Type" content="text/html; charset=ISO-8859-1">
<title>Insert title here</title>
</head>
<body style="background-color:#FFEBCD;">
<%@ page import="java.sql.*"%>
<br/>
<br/>
<img src ="img/ad.jpg" width="150px" height="150px" class="img-circle">

<h><font size="14color="#000000">Admindashboard</font></h>
<br/>

 <div class="container">

<div class="row">

<div class="col-md-5">
<%@ include file="adminleftmenu.jsp" %>
<br/>
</div>
<div class="col-md-1">

</div>
<div class="col-md-6">
<br/>
   <font size="5"> " You may wish to block words that are often found in comment
spam or offensive words you do not want published to your blog.
It is important to note that certain words that are often used
in spamcomments might also be used legitimately. For example,you
may wish to block a word like "poker" to avoid spam of that nature.
However, you should also consider whether a legitimate reader might 
want to discuss poker in a comment."</font>
<br/>
</div>
 </div>
 </div>
</body>
</html>'''
    st.markdown(page_bg_img,unsafe_allow_html=True)
    st.header("TEMPLATES SUGGESTION ML App")
    utt = st.text_input( "USER TEXT","")
  
    result=""
    if st.button("MAIL TEMPLATE"):
        result=classify_utterance(utt)
    
        st.success('{}'.format(result))
   
if __name__=='__main__':
    main()
    
    
