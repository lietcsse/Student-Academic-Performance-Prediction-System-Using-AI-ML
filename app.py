import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore
import google.generativeai as genai
import os
from dotenv import load_dotenv
from flask import Flask, request, jsonify,render_template,redirect,url_for
from flask_cors import CORS
import numpy as np
os.environ['MPLBACKEND'] = 'agg'
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from waitress import serve




app = Flask(__name__)
CORS(app)
load_dotenv()

# Configure Flask to serve static files from the 'static' directory
app.static_url_path = '/static'
app.static_folder = 'static'


# Generate a random string of 32 characters for the secret key
secret_key = os.urandom(32)
app.secret_key = secret_key

# Configure Gemini API key
gemini_api_key = os.getenv("GEMINI_API_KEY")
if not gemini_api_key:
    raise ValueError("GEMINI_API_KEY environment variable not set")
genai.configure(api_key=gemini_api_key)

# Initialize Firebase Admin SDK
if not firebase_admin._apps:
    cred = credentials.Certificate(r"C:\Users\kodad\OneDrive\Desktop\Student_Performance_Prediction\config_key.json")
    firebase_admin.initialize_app(cred)
db = firestore.client()

# Function to retrieve selected subjects from Firestore
def get_selected_subjects(user_id):
    try:
        user_ref = db.collection('users').document(user_id)
        user_doc = user_ref.get()

        if user_doc.exists:
            selected_subjects = user_doc.get('selectedSubjects')
            if selected_subjects:
                selected_subjects_list = [subject for subject in selected_subjects]
                return selected_subjects_list
            else:
                return []
        else:
            print(f"User document with ID {user_id} does not exist.")
            return []
    except Exception as e:
        print(f"Error retrieving selected subjects: {e}")
        return []

# Function to retrieve Student Cgpa from Firebase
def get_selected_Cgpa(user_id):
    try:
        user_ref = db.collection('users').document(user_id)
        user_doc = user_ref.get()

        if user_doc.exists:
            selected_cgpa = user_doc.get('Cgpa')
            if selected_cgpa:
                selected_cgpa_list = [float(item['cgpa']) for item in selected_cgpa if item.get('cgpa') != ''] 
                return selected_cgpa_list
            else:
                return []
        else:
            print(f"User document with ID {user_id} does not exist.")
            return []
    except Exception as e:
        print(f"Error retrieving cgpa: {e}")
        return []
    
# Function to predict the future cgpa
def predict_future_cgpa(selected_cgpa_list):
    num_semesters = len(selected_cgpa_list)
    X = np.arange(1, num_semesters + 1).reshape(-1, 1)
    y = np.array(selected_cgpa_list)

    model = LinearRegression()
    model.fit(X, y)

    future_semesters = np.arange(num_semesters + 1, min(num_semesters + 6, 9)).reshape(-1, 1)
    predicted_cgpa = model.predict(future_semesters)
    predicted_cgpa = np.minimum(predicted_cgpa, 10)
    return predicted_cgpa

# Function to retrieve other info  of a student 
def retrieve_other_info(user_id):
        try:
             user_ref = db.collection('users').document(user_id)
             user_doc = user_ref.get()

             if user_doc.exists:
                 other_info={}
                 data=user_doc.to_dict()
                 if 'sleepingHours' in data:
                    other_info['sleepingHours'] = data['sleepingHours']
                 if 'studyHours' in data:
                    other_info['studyHours'] = data['studyHours']
                 if 'screenTime' in data:
                    other_info['screenTime'] = data['screenTime']
                 if 'learningStyle' in data:
                    other_info['learningStyle'] = data['learningStyle']

                 return other_info
             else :
                 return f'doc {user_id} does not exist'
        except Exception as e:
            print(f"Error retreiving other info: {e}")     


# Route to handle user requests
@app.route('/user', methods=['POST'])
def receive_user_id():
    try:
        user_id = request.json.get('userId')
        if user_id:
            output = get_selected_subjects(user_id)
            piechart=retrieve_other_info(user_id)
            if output and piechart is not None:
                piechart_data = {key: value for key, value in piechart.items() if key != 'learningStyle'}
                total_hours = 24
                screen_time = int(piechart_data.get('screenTime', 0))
                study_time = int(piechart_data.get('studyHours', 0))
                sleep_hours = int(piechart_data.get('sleepingHours', 0))
                other_activities_hours = total_hours - (screen_time + study_time + sleep_hours)
                if other_activities_hours<0:
                    other_activities_hours=0
                    piechart_data['Other Activities'] = other_activities_hours
                kt_tips = generate_output(output)
                comparison=generate_comparison(piechart_data)
                create_piechart(user_id,piechart_data)
                if kt_tips and comparison :
                    user_ref = db.collection('users').document(user_id)
                    user_ref.update({'kt_tips': kt_tips,'comparison': comparison})
                    return redirect(url_for('display_kt_output',user_id=user_id))
                else:
                    return 'Error in generating LLM response', 200
            else:
                return 'User data not found', 404
        else:
            return 'User ID not provided', 400
    except Exception as e:
        print('Error:', e)
        return 'Internal Server Error', 500

import re

# Function to remove unwanted markdown/HTML formatting
def remove_formatting(text):
    # Remove markdown or HTML tags (if present)
    text = re.sub(r'[*_]', '', text)  # Removes * and _ (used in markdown for bold/italics)
    text = re.sub(r'<[^>]*>', '', text)  # Removes HTML tags
    return text

# Function to generate KT-tips with an Indian college focus, including YouTube resources and key topics
def generate_output(selected_subjects_list):
    try:
        model = genai.GenerativeModel("gemini-1.5-pro")
        
        # Make sure the prompt is adjusted to prevent any bold characters and avoid empty semester messages
        prompt_2 = f"""Semester and Subjects: {selected_subjects_list}

Goal: Provide effective, exam-oriented strategies for improving academic performance in the listed subjects, tailored for Indian college students.

Instructions:

- Keep the points short, clear, and practical.
- Keep the strategies concise and practical, avoiding any special formatting, bold text, or bullet points with asterisks.
- Focus on:
  - Key Topics: Include all main topics (more than 10 topics per subject).
  - Best learning resources (NPTEL, YouTube channels, university lecture notes).
  - Standard textbooks used in Indian universities.
  - Previous year questions and common exam patterns.
  - Quick revision strategies (mind maps, one-page summaries, formula sheets).
  - Time management techniques for balancing multiple subjects.
  - Recommended mobile apps for doubt solving and self-tests.
  - Best YouTube Channels for learning in ENGLISH, TELUGU, HINDI separately
  -dont need space between lines

Example output format (without stars or extra symbols):
Engineering Mathematics 2 (SEM 2)
Key Topics: Laplace Transform, Differential Equations, Vector Calculus, Fourier Series, Eigenvalues and Eigenvectors, Complex Analysis, Numerical Methods, Linear Algebra, Partial Differential Equations, Systems of Linear Equations, Integral Calculus
Resources: NPTEL lectures by IIT Professors, GATE Academy YouTube videos
Textbooks: Higher Engineering Mathematics by B.S. Grewal, Advanced Engineering Mathematics by Erwin Kreyszig
Exam Pattern: Focus on university past papers, commonly repeated questions
Revision: Solve previous year questions, create a formula sheet for quick reference
Time Management: Practice numerical problems daily, allocate 2 hours per subject
YouTube Channels: 
    English - NPTEL, GATE Academy
    Telugu - Telugu Academy, Edureka Telugu
    Hindi - Examrace Hindi, GATE Academy Hindi

Ensure that the output follows this plain text format and does not contain any stars, asterisks, or the statement about empty subjects."""

        response_2 = model.generate_content(prompt_2)
        
        # Post-process the response to ensure no unwanted statements or formatting
        output = response_2.text
        
        # Remove any unwanted statements about empty subjects (if they exist in the response)
        output = output.replace("Subjects for remaining semesters are currently empty.", "").strip()
        
        # Remove markdown or HTML formatting
        output = remove_formatting(output)
        
        return output
    except Exception as e:
        return f"Error generating response: {e}"



import re

# Function to remove unwanted formatting from the text
def remove_formatting(text):
    # Remove markdown or HTML tags (if present)
    text = re.sub(r'[*_]', '', text)  # Removes * and _ (used in markdown for bold/italics)
    text = re.sub(r'<[^>]*>', '', text)  # Removes HTML tags
    return text

# Function to generate personalized and motivating schedule comparison with actionable tips
def generate_comparison(piechart_data):
    try:
        model = genai.GenerativeModel("gemini-1.5-pro")
        ideal_data = {'Max_ScreenTime': 3, 'Enough_SleepHours': 8, 'Min-StudyTime': 2, 'Other_Activities': 12}
        
        # Updated prompt to avoid stars, bold formatting, and excessive details
        prompt_2 = f"""Task:first writecomparing values,then Compare a student's current daily schedule with an ideal daily schedule and provide strategies to help them align with the ideal small and practical points only, dont compare it in table.

Instructions:
Imagine you are a personal guide mentoring the student.
- Your task is to offer practical, actionable advice to help the student gradually move toward the ideal schedule.
- Focus on making the adjustments achievable, sustainable, and realistic.
- Be clear, concise, and actionable, while maintaining a supportive tone.
-dont need space between lines

Guidance should include:
- Practical steps for improving study habits.
- Tips for managing time effectively between study and other activities.
- Suggestions for balancing screen time and ensuring enough sleep.
- Simple, achievable changes that can help in aligning with the ideal schedule.
- Try to say the tips in a positive way.


Student's Current Daily Schedule: ({piechart_data})
Ideal Daily Schedule: ({ideal_data})

Provide your guidance to help the student make incremental changes toward the ideal schedule. No side headings needed. Generate without bolding the texts. Tips should be concise, actionable, and in a supportive tone.
"""

        # Get model's response
        response = model.generate_content(prompt_2)
        
        # Remove unwanted formatting from the response
        cleaned_response = remove_formatting(response.text)
        
        # Return the cleaned response
        return cleaned_response
    except Exception as e:
        error_message = f"Error generating response: {e}"
        return error_message



# function to generate schedule piechart
def create_piechart(user_id,piechart_data):
    labels = list(piechart_data.keys())
    values = list(piechart_data.values())
    colors = ['gold', 'yellowgreen', 'lightcoral', 'lightskyblue']
    plt.switch_backend('agg')
    plt.pie(values, labels=labels,colors=colors, autopct='%1.1f%%', startangle=140)
    plt.title('PIECHART OF YOUR DAILY SCHEDULE')
    plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
    piechart_filename=f"static/{user_id}_performance_piechart.png"
    plt.savefig(piechart_filename)
    plt.clf()    
    return 'Piechart saved'

# Function to handle cgpa request
@app.route('/cgpa', methods=['POST'])
def receive_user_id_cgpa():
    try:
        user_id = request.json.get('userId')
        if user_id:
            output = get_selected_Cgpa(user_id)
            piechart=retrieve_other_info(user_id)
            if output and piechart is not None:
                piechart_data = {key: value for key, value in piechart.items() if key != 'learningStyle'}
                total_hours = 24
                screen_time = int(piechart_data.get('screenTime', 0))
                study_time = int(piechart_data.get('studyHours', 0))
                sleep_hours = int(piechart_data.get('sleepingHours', 0))
                other_activities_hours = total_hours - (screen_time + study_time + sleep_hours)
                if other_activities_hours<0:
                    other_activities_hours=0
                    piechart_data['Other Activities'] = other_activities_hours
                cgpa = predict_future_cgpa(output)
                if cgpa is not None:
                   tips=generate_tips(output,cgpa)
                   create_nonKT_piechart(user_id,piechart_data)
                   Create_Graph(output,cgpa,user_id)
                   comparison=generate_comparison(piechart_data)
                   if user_id and tips and comparison:
                       user_ref = db.collection('users').document(user_id)
                       user_ref.update({'tips': tips,'comparison': comparison})
                       return redirect(url_for('display_output',user_id=user_id))
                   else:
                        return 'User ID or tips not generated', 500
                else:
                    return 'Error in generating Graph', 200
            else:
                return 'User data not found', 404
        else:
            return 'User ID not provided', 400
    except Exception as e:
        print('Error:', e)
        return 'Internal Server Error', 500



import re
# Function to generate graph trend tips
def generate_tips(selected_cgpa_list, predicted_cgpa):
    try:
        model = genai.GenerativeModel("gemini-1.5-pro")
        prompt_2 = f"""Given your historical CGPA records:({selected_cgpa_list}) and the predicted future CGPAs:({predicted_cgpa}), provide tailored strategies to either address a declining trend or capitalize on an inclining trend in academic performance (only one) compulsorily checking only from left to right in the given data collectively. Ensure the output offers clear and actionable small bullet points advices suitable for the identified trend, maintaining a professional and easily understandable structure in small bulleted points(consised and informative) with space between them.)"""

        response_2 = model.generate_content(prompt_2)
        clean_response = remove_formatting(response_2.text.strip())
        return clean_response
    except Exception as e:
        error_message = f"Error generating response: {e}"
        return error_message

# Helper function to remove any markdown or HTML formatting
def remove_formatting(text):
    # Remove markdown or HTML tags (if present)
    text = re.sub(r'[*_`]', '', text)  # Removes * and _ (used in markdown for bold/italics)
    text = re.sub(r'<[^>]*>', '', text)  # Removes HTML tags
    return text


    
# Function to create a schedule piechart
def create_nonKT_piechart(user_id,piechart_data):
    labels = list(piechart_data.keys())
    values = list(piechart_data.values())
    colors = ['gold', 'yellowgreen', 'lightcoral', 'lightskyblue']
    plt.switch_backend('agg')
    plt.pie(values, labels=labels,colors=colors, autopct='%1.1f%%', startangle=140)
    plt.title('PIECHART OF YOUR DAILY SCHEDULE')
    plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
    piechart_filename=f"static/{user_id}_performance_piechart.png"
    plt.savefig(piechart_filename)
    plt.clf()    
    print('Piechart saved')

# Function to create a prediction graph
def Create_Graph(output,cgpa,user_id):
     num_semesters = len(output)
     future_semesters = np.arange(num_semesters + 1, min(num_semesters + 6, 9))
     plt.switch_backend('agg')
     plt.figure(figsize=(8, 6))
     plt.plot(future_semesters, cgpa, linestyle='dashed', color='green')  # Line for predicted CGPAs
     plt.scatter(np.arange(1, num_semesters + 1), output, color='blue', label='Past CGPA')
     plt.scatter(future_semesters, cgpa, color='green', label='Predicted CGPA')
     plt.title('CGPA Prediction for Future Semesters')
     plt.xlabel('Semester Number')
     plt.ylabel('CGPA')
     plt.xlim(0,9)
     plt.ylim(0, 11)
     plt.subplots_adjust(top=0.9,bottom=0.1)
     plt.legend()
     plt.grid(True)
     filename=f"static/{user_id}_performance_graph.png"
     plt.savefig(filename)
     plt.clf()
     print("Image Saved Successfully!")


import re

# Function to remove formatting from the text
def remove_formatting(text):
    # Remove markdown or HTML tags (if present)
    text = re.sub(r'[*_`]', '', text)  # Removes * and _ (used in markdown for bold/italics)
    text = re.sub(r'<[^>]*>', '', text)  # Removes HTML tags
    return text

# Function to generate schedule comparison with actionable strategies
def generate_comparison(piechart_data):
    try:
        model = genai.GenerativeModel("gemini-1.5-pro")
        ideal_data = {'Max_ScreenTime': 2, 'Enough_SleepHours': 8, 'Min-StudyTime': 2, 'Other_Activities': 12}
        
        # Creating the prompt with simplified and concise strategies
        prompt_2 = f"""Task: Compare a student's current daily schedule with an ideal daily schedule and provide strategies to align them.

Instructions:

Given the current daily schedule of the student and an ideal daily schedule, provide simple and actionable strategies in 3 to 4 lined bulleted points to help the student adjust their schedule towards the ideal one.
Keep the strategies concise and practical. show it with proper alignmnent and spacing.

Prompt:
"Student's Current Daily Schedule: ({piechart_data})
Ideal Daily Schedule: ({ideal_data})

Compare the student's current daily schedule with the ideal daily schedule and provide strategies. Offer actionable recommendations to help the student align their schedule with the ideal.
Provide brief, actionable advice with practical steps the student can implement daily."""

        # Get model's response
        response = model.generate_content(prompt_2)
        
        # Clean up the response by removing formatting
        clean_response = remove_formatting(response.text)
        
        # Process and return the cleaned-up response
        return clean_response
    except Exception as e:
        error_message = f"Error generating response: {e}"
        return error_message

       
# function to handle output request and render the output page
@app.route('/output/<string:user_id>')
def display_output(user_id):
       return render_template("output.html", user_id=user_id)

# funtion to handle kt-output request and render the kt-output page
@app.route('/kt_output/<string:user_id>')
def display_kt_output(user_id):
    return render_template("kt-output.html", user_id=user_id)


def main():
   host = os.getenv('HOST', '0.0.0.0')
   port = int(os.getenv('PORT', '5000'))  # Convert port to integer
   serve(app, host=host, port=port)
    

# Start Flask app
if __name__ == '__main__':
    app.run(debug=True)
  
    