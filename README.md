# **Resume & Job Application Advisor System**


An **agentic system powered by CrewAI and open-source LLMs** that helps job seekers improve their resumes, discover current job openings, generate tailored cover letters, and prepare for interviews — all in a single interactive **Streamlit app**.  

✨ Designed to be efficient, creative (but grounded), and cost-free by leveraging open-source models and APIs.

---

## Features

1. Upload your resume (.txt).  
2. Specify your professional field and country.  
3. Get:
- Resume feedback & ranking
- Resume improvement suggestions
- Current job openings relevant to your field & skills
- Customized resume tailored to openings
- Draft cover letters for each job
- Interview preparation (likely questions & model answers)

---

## Workflow
- create the project environment: with `bash: conda create -n 'env_name' python=3.11` followed by the `bash: conda activate 'env_name'`

The system follows this **7-step workflow**:

1. **Setup requirements & environment**
- Read API keys from `crewai_api_key.yml` and `huggingface_credentials.yml`, alternatively using `openai_credentials.yml` for the `gpt-4o`
- Initialize the open-source LLM (Mistral-7B-Instruct-v0.3 via HuggingFace)

2. **Import important libraries**

3. **Setup project environment**
- Cache the LLM to minimize latency
- Detect GPU if available

4. **Prepare the agents**
- 6 Agents:
  - Resume Feedback Agent
  - Resume Advisor Agent
  - Job Researcher Agent
  - Resume Customizer Agent
  - Cover Letter Agent
  - Interview Coach Agent

5. **Create the CrewAI tasks**
- Assign each task to the respective agent

6. **Get the outputs**
- Execute the CrewAI pipeline using: '*streamlit run app.py*'
- Display results in clear sections

7. **Deploy using Streamlit**
- Sidebar inputs & interactive interface

---

## Tech Stack

- [Streamlit](https://streamlit.io/) for web UI
- [CrewAI](https://crewai.com/) for agent orchestration
- [HuggingFace Transformers](https://huggingface.co/docs/transformers) for open-source LLM
- [SERPER API](https://serper.dev/) for live job search results
- PyYAML & Requests for config & HTTP

---

## Installation & Setup

### Clone the repository
```bash
git clone https://github.com/your-username/resume-agentic-system.git
cd resume-agentic-system
