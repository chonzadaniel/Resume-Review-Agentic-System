# --------------------------------------------- Install and import Libraries --------------------------------
import warnings
warnings.filterwarnings('ignore')

import os
import io
import yaml
import streamlit as st
from datetime import datetime, timedelta
from PyPDF2 import PdfReader
from docx import Document
from crewai import Agent, Task, Crew, Process
from crewai_tools import SerperDevTool
from openai import OpenAI


# -------------------------------------- Setup requirements — environment & API keys ----------------------------
def load_api_keys():
    with open("openai_credentials.yml") as f:
        openai_keys = yaml.safe_load(f)
    os.environ["OPENAI_API_KEY"] = openai_keys.get("OPENAI_API_KEY", "")

    if os.path.exists("crewai_api_key.yml"):
        with open("crewai_api_key.yml") as f:
            crewai_keys = yaml.safe_load(f)
        os.environ["SERPER_API_KEY"] = crewai_keys.get("SERPER_API_KEY", "")

load_api_keys()

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
SERPER_API_KEY = os.environ.get("SERPER_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY is missing! Please check your openai_credentials.yml file.")

# ----------------------------------------------- Streamlit UI ----------------------------------------
st.set_page_config(page_title="Resume & Job Application Advisor", layout="wide")
st.title("📄 **Resume & Job Application Advisor**")

with st.sidebar:
    st.header("User Input")
    uploaded_resume = st.file_uploader(
        "Upload your Resume (.txt, .pdf, .docx)",
        type=["txt", "pdf", "docx"]
    )
    professional_field = st.text_input(
        "Your Professional Field",
        placeholder="e.g., Data Science"
    )
    country = st.text_input(
        "Country",
        placeholder="e.g., Tanzania"
    )
    run_button = st.button("Run Agentic System")

# --------------------------------------- Resume Extraction -----------------------------------------------
def read_resume(uploaded_file):
    if uploaded_file.name.endswith(".txt"):
        return uploaded_file.read().decode("utf-8", errors="ignore")
    elif uploaded_file.name.endswith(".pdf"):
        pdf_reader = PdfReader(uploaded_file)
        text = "".join(page.extract_text() or "" for page in pdf_reader.pages)
        return text.strip()
    elif uploaded_file.name.endswith(".docx"):
        docx_file = io.BytesIO(uploaded_file.read())
        document = Document(docx_file)
        text = "\n".join(para.text for para in document.paragraphs)
        return text.strip()
    else:
        raise ValueError("Unsupported file type")

# -------------------------------------------- Agents & Tasks ------------------------------------------
search_tool = SerperDevTool()

today = datetime.today()
min_deadline = (today + timedelta(days=3)).strftime('%d %B %Y')

def create_agents_and_tasks(resume_text, professional_field, country):
    resume_feedback = Agent(
        name="Resume Feedback Agent",
        role="Professional Resume Reviewer",
        goal="Give critical feedback and rate the resume between 0–10.",
        backstory="An experienced career consultant skilled in analyzing resumes.",
        llm="gpt-4o"
    )

    resume_advisor = Agent(
        name="Resume Advisor Agent",
        role="Professional Resume Writer",
        goal="Rewrite the resume to maximize attractiveness to recruiters.",
        backstory="An expert resume writer crafting compelling narratives.",
        llm="gpt-4o"
    )

    job_researcher = Agent(
        name="Job Researcher Agent",
        role="Senior Recruitment Consultant",
        goal="Find the top 3 relevant jobs with deadlines on or after the next 3 days.",
        backstory="An expert recruiter finding relevant job opportunities.",
        tools=[search_tool],
        llm="gpt-4o"
    )

    resume_customizer = Agent(
        name="Resume Customizer Agent",
        role="Resume Customization Specialist",
        goal="Tailor the improved resume to each job posting powerful enough to convince employers to hire the candidate.",
        backstory="An expert in optimizing resumes to match job descriptions.",
        llm="gpt-4o"
    )

    job_applicant = Agent(
        name="Cover Letter Writer Agent",
        role="Senior Job Application Consultant",
        goal="Write tailored cover letters for each job posting demonstrating suitability of candidacy.",
        backstory="A skilled cover letter strategist.",
        llm="gpt-4o"
    )

    interview_coach = Agent(
        name="Interview Coach Agent",
        role="Job Interview Coach",
        goal="Prepare the applicant with potential questions & answers for each job opening with a practical eye.",
        backstory="An experienced interview coach helping candidates succeed securing jobs for suitability to open jobs comprehensively.",
        llm="gpt-4o"
    )

    tasks = [
        Task(
            description="Evaluate the resume, rate it out of 10, and provide detailed feedback in markdown.",
            expected_output="Score should be in a range of (0–10) and feedback in markdown format spotting key issues for improvement by the resume advisor.",
            agent=resume_feedback
        ),
        Task(
            description="""Rewrite the resume in markdown format based on the feedback aimed at improving the candidate's suitability to open jobs. 
                           Ensure the original resume facts about the owner employment records maintained (no adding unrealistic information) for genuinity and trustworthy.
                        """,
            expected_output="Improved resume in markdown format.",
            agent=resume_advisor
        ),
        Task(
            description=f"Find the top 5 jobs in {country} matching {professional_field} with deadlines on or after {min_deadline}, and output in markdown.",
            expected_output="Markdown list of up to 3 job postings with details.",
            agent=job_researcher
        ),
        Task(
            description="Customize the improved resume for each of the 3 job postings.",
            expected_output="Tailored resumes for each job in markdown format ensuring the original career history records not changed, rather improved for suitability to the jobs.",
            agent=resume_customizer
        ),
        Task(
            description="Write tailored cover letters for each of the 3 job postings.",
            expected_output="Cover letters for each job in markdown format bolding the suitability of the candidacy to open jobs, well customized for a good fit.",
            agent=job_applicant
        ),
        Task(
            description="Prepare interview questions and answers based on the tailored resumes and jobs in markdown.",
            expected_output="Interview preparation guiding questions and respective answers in markdown format comprehensively detailed as possible.",
            agent=interview_coach
        )
    ]

    agents = [
        resume_feedback, resume_advisor, job_researcher,
        resume_customizer, job_applicant, interview_coach
    ]

    return agents, tasks

# --------------------------------------- Run Agentic System ---------------------------------------
if run_button and uploaded_resume:
    with st.spinner("📄 Reading your resume..."):
        try:
            resume_text = read_resume(uploaded_resume)
        except Exception as e:
            st.error(f"Failed to read resume: {e}")
            st.stop()

    st.subheader("Results")

    pf = professional_field.strip() or "General"
    loc = country.strip() or "Any"

    agents, tasks = create_agents_and_tasks(resume_text, pf, loc)

    crew = Crew(
        agents=agents,
        tasks=tasks,
        process=Process.sequential,
        verbose=True
    )

    with st.spinner("Agents are working on your request..."):
        results = crew.kickoff(inputs={"resume": resume_text, "location": loc})

    sections = [
        "Resume Feedback & Rating",
        "Improved Resume",
        "Top 5 Current Job Openings",
        "Tailored Resumes",
        "Sample Cover Letters",
        "Interview Preparation Guide"
    ]

    for i, result in enumerate(results):
        st.markdown(f"## {sections[i]}")

        # Cleanly extract string content from result
        if hasattr(result, 'raw') and result.raw:
            output_text = result.raw
        elif hasattr(result, 'content') and result.content:
            output_text = result.content
        elif isinstance(result, str):
            output_text = result
        else:
            output_text = str(result)

        # Remove unintended escape characters
        cleaned_text = output_text.replace('\\n', '\n').replace('\\t', '\t')

        # Optional: if the agent accidentally wrapped markdown in code fences, strip them
        if cleaned_text.startswith('```') and cleaned_text.endswith('```'):
            cleaned_text = cleaned_text.strip('`').strip()

        # Render as markdown
        st.markdown(cleaned_text, unsafe_allow_html=True)


elif run_button:
    st.error("🚨 Please upload your resume before running the Agentic System.")
