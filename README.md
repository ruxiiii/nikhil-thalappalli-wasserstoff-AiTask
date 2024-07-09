# WordPressQABot

## Chat with Your WordPress Blogs
This application allows users to interact with their WordPress blogs through a chatbot interface. The chatbot extracts content from the provided WordPress blog URL, processes it, and answers user questions based on the relevant excerpts from the blog content.

## Table of Contents
* Installation
* Usage
* File Structure
* Function Overview
* Environment Variables
* Running the Application
* Dependencies


## Installation :

1. Clone the repository:
```
  git clone <repository_url>
  cd <repository_directory>
```

2. Create and activate virtual environment :
```
  python3 -m venv venv
  venv/Scripts/Activate.ps1
```

3. Install required packages:
```
  pip install -r requirements.txt
```

## Environment Variables :

GROQ_API_KEY=your_groq_api_key


## Usage : 

Run the application :

```
  streamlit run app.py
```

## File Structure : 
.
├── app.py              
├── requirements.txt     
├── .env                
├── base_prompt.txt     
└── htmlTemplates.py  




  
