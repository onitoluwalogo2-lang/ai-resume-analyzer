# Imports

import json
import os
import re
import pymupdf
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI
from reference_jobs import jobs


# OpenAI Setup

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=api_key)

print("OpenAI client created successfully")

# Resume Extraction

def extract_candidate_profile(resume_text):
    response = client.responses.create(
        model="gpt-5-mini",
        input=f"""
Analyze the following resume and extract the candidate's information.

Return ONLY valid JSON in this exact structure:

{{
    "skills": [],
    "projects": [],
    "education": [],
    "experience": []
}}

Rules for skills:
- Extract explicit technical skills, tools, technologies, platforms, programming languages, APIs, AI/ML competencies, and professional competencies demonstrated by the candidate.
- Extract skills from the Technical Skills section, Professional Summary, Projects, AI Training, Career Focus, and relevant project descriptions when the skill is clearly demonstrated.
- Include AI Automation when the resume explicitly mentions AI automation, automation, automated workflows, or automation as a demonstrated skill or career focus.
- Include LLMs when the resume mentions LLMs, Large Language Models, or Large Language Model fundamentals.
- Include RAG when the resume mentions RAG or Retrieval-Augmented Generation.
- Include Prompt Engineering when explicitly demonstrated or listed.
- Include Embeddings when explicitly demonstrated or listed.
- Include Semantic Search when explicitly demonstrated or listed.
- Include AI Agents when explicitly demonstrated or listed.
- Include LLM Evaluation when explicitly demonstrated or listed.
- Include LLMOps when explicitly demonstrated or listed.
- Include Guardrails when explicitly demonstrated or listed.
- Include Red Teaming when explicitly demonstrated or listed.
- Include API-related skills such as OpenAI API, API Integration, REST APIs, or API Testing when clearly demonstrated.
- Include Git and GitHub when clearly listed or demonstrated.
- Do NOT include personality traits such as hardworking, motivated, organized, creative, analytical, etc.
- Do NOT invent skills that are not supported by the resume.
- Do NOT treat a job title alone as proof of a skill.
- Keep multi-word skills as complete phrases.
- Avoid duplicate skills.
- If the same skill appears in different forms, use one clear standardized version.

Rules for projects:
- Extract actual projects listed in the resume.
- Include the project name.
- Include a concise description based only on the resume.
- Include technologies explicitly associated with the project.
- Include responsibilities or actions explicitly demonstrated in the project.
- Do not invent project outcomes.

Rules for education:
- Extract the institution, degree/program, and graduation year when available.

Rules for experience:
- Extract actual professional employment experience only.
- Do not treat personal projects, courses, training, or education as professional employment.
- If no professional employment experience is listed, return an empty list.

Resume:
{resume_text}
"""
    )

    return response.output_text

   # PDF Text Extraction

def extract_text_from_pdf(pdf_path):
    document = pymupdf.open(pdf_path)

    text = ""

    for page in document:
        text += page.get_text()

    document.close()

    return text


resume_text = extract_text_from_pdf("resume.pdf")


profile = extract_candidate_profile(resume_text)

# Job Extraction

def extract_job_profile(job_text):

    response = client.responses.create(
        model="gpt-4o-mini",
        input=f"""
You are an AI job-profile extraction system.

Analyze the COMPLETE job posting below.

Return ONLY valid JSON.
Do not return markdown.
Do not return explanations.

Use exactly this structure:

{{
    "title": "",
    "company": "",
    "required_skills": [],
    "experience_required": 0,
    "description": "",
    "responsibilities": []
}}

REQUIRED_SKILLS RULES:

- Extract explicit skills, tools, technologies, frameworks,
  platforms, programming languages, APIs, and professional competencies.
- Include skills explicitly required or expected to perform the job.
- Check sections such as:
  Requirements
  Required Qualifications
  Qualifications
  Technical Skills
  Must Have
  Preferred Qualifications
  Skills
- Include "automation" when the posting explicitly mentions:
  automation, AI automation, workflow automation,
  process automation, automated workflows, or automation workflows.
- Include "API Integration" when the posting explicitly mentions:
  API integration, integrating APIs, consuming APIs,
  connecting external services, or system integrations.
- Include "REST APIs" when REST API or RESTful API work is explicitly required.
- Include "error handling" when error handling, exception handling,
  fault handling, or resilient application behavior is explicitly required.
- Include "API testing" when API testing, endpoint testing,
  integration testing, or automated API testing is explicitly required.
- Keep multi-word skills as complete phrases.
- Do not include personality traits.
- Do not include vague characteristics.
- Do not convert ordinary responsibilities into skills unless
  they explicitly name a technical skill or competency.
- Do not invent skills that are not present in the job posting.

EXPERIENCE RULES:

- Extract the number of years of professional experience explicitly required.
- Examples:
  "2+ years" = 2
  "3 years of experience" = 3
  "1-2 years" = 1
- If no professional experience requirement is stated, use 0.

DESCRIPTION RULES:

- Provide a concise summary of what the job is about.

RESPONSIBILITIES RULES:

- Extract the actual duties and responsibilities.
- Keep responsibilities as duties.
- Do not convert responsibilities into required_skills.

IMPORTANT EXTRACTION PROCESS:

1. Read the entire job posting.
2. Identify the job title.
3. Identify the company if provided.
4. Identify explicit required skills.
5. Identify the experience requirement.
6. Summarize the job.
7. Extract the actual responsibilities.
8. Return the extracted information as JSON.

If information is not present, use an empty string, empty list,
or 0 as appropriate.

Return ONLY the JSON object.

JOB POSTING:

{job_text}
"""
    )

    return response.output_text

profile_data = json.loads(profile)


print("\nPaste the job posting below.")
print("When you're finished, type END on a new line and press Enter.")

job_lines = []

while True:
    line = input()

    if line.strip() == "END":
        break

    job_lines.append(line)

job_posting_text = "\n".join(job_lines)

job_profile = extract_job_profile(job_posting_text)

job_data = json.loads(job_profile)
skills = profile_data["skills"]

skills_normalized = {
    skill.strip().lower()
    for skill in skills
}

projects = profile_data["projects"]
education = profile_data["education"]
experience = profile_data["experience"]

project_count = len(projects)


# Similarity Calculation

def calculate_similarity(vector_a, vector_b):
    dot_product = np.dot(vector_a, vector_b)

    magnitude_a = np.linalg.norm(vector_a)
    magnitude_b = np.linalg.norm(vector_b)

    return dot_product / (magnitude_a * magnitude_b)

    # Embeddings

def create_embedding(text):
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )

    return response.data[0].embedding

# Job Embeddings

def create_job_embedding(job):

    job_text = (
        "Job Title: " + job["title"]
        + "\nRequired Skills: "
        + ", ".join(job["required_skills"])
        + "\nDescription: "
        + job.get("description", "")
        + "\nResponsibilities: "
        + " ".join(job.get("responsibilities", []))
    )

    return create_embedding(job_text)

# Reference jobs used to benchmark the candidate against example roles

candidate_experience = sum(
    float(item.get("years", 0) or 0)
    for item in experience
)

experience_penalty = 0.20
PROJECT_RELEVANCE_THRESHOLD = 0.50

# Candidate Background

project_text = ""

for project in projects:
    project_text += (
    "\nProject: " + project.get("name", "")
    + "\nTechnologies: " + ", ".join(project.get("technologies", []))
    + "\nDescription: " + str(project.get("description", ""))
    + "\nResponsibilities: " + " ".join(project.get("responsibilities", []))
    + "\nOutcomes: " + " ".join(project.get("outcomes", []))
)

education_text = ""

for item in education:
    if isinstance(item, dict):
        education_text += (
            "\nInstitution: " + str(item.get("institution", ""))
            + "\nDegree: " + str(item.get("degree", ""))
            + "\nGraduation year: " + str(item.get("graduation_year", ""))
        )
    else:
        education_text += "\nEducation: " + str(item)

# Experience Data

experience_text = ""

for item in experience:
    experience_text += (
        "\nJob Title: " + item.get("job_title", "")
        + "\nCompany: " + item.get("company", "")
        + "\nStart Date: " + item.get("start_date", "")
        + "\nEnd Date: " + item.get("end_date", "")
        + "\nYears: " + str(item.get("years", 0))
    )
# Resume Embedding Text

resume_text_for_embedding = (
    "Skills: " + ", ".join(skills)
    + project_text
    + education_text
    + experience_text
)

# Candidate Background Embedding

candidate_background_text = (
    project_text
    + experience_text
)

candidate_background_embedding = create_embedding(
    candidate_background_text
)

resume_embedding = create_embedding(resume_text_for_embedding)

job_results = []

# Skill Matching

SKILL_ALIASES = {
    "llms": [
        "large language models",
        "large language model"
    ],

    "large language models": [
        "llms"
    ],

    "rag": [
        "retrieval-augmented generation",
        "retrieval augmented generation"
    ],

    "retrieval-augmented generation": [
        "rag"
    ],

    "apis": [
        "api",
        "api integration",
        "rest api",
        "rest apis"
    ],

    "api": [
        "apis",
        "api integration",
        "rest api",
        "rest apis"
    ],

    "api integration": [
        "api",
        "apis",
        "rest api",
        "rest apis"
    ],

    "rest api": [
        "api",
        "apis",
        "api integration",
        "rest apis"
    ],

    "rest apis": [
        "api",
        "apis",
        "api integration",
        "rest api"
    ],

    "automation": [
        "ai automation"
    ],

    "ai automation": [
        "automation"
    ],

    "ai agents": [
        "agents"
    ],

    "agents": [
        "ai agents"
    ],

    "machine learning": [
        "ml",
        "machine-learning"
    ],

    "ml": [
        "machine learning"
    ],

    "numpy": [
        "numerical python"
    ],

    "pandas": [
        "python pandas"
    ],

    "sql": [
        "structured query language"
    ],

    "structured query language": [
        "sql"
    ]
}

# Skill Recommendations

SKILL_RECOMMENDATIONS = {
    "python": "Practice Python by building small AI applications and automation scripts.",

    "llms": "Learn how large language models work and practice building LLM-powered applications.",

    "rag": "Learn Retrieval-Augmented Generation and build a small RAG application.",

    "embeddings": "Learn how embeddings work and practice using them for semantic search.",

    "sql": "Learn SQL fundamentals and practice querying and managing databases.",

    "apis": "Learn how APIs work and practice connecting AI applications to external services.",

    "automation": "Practice building AI automation workflows that connect different tools and services.",

    "ai agents": "Learn agentic workflows and practice building AI agents that can use tools.",

    "digital marketing": "Learn digital marketing fundamentals and build a practical campaign project.",

    "social media marketing": "Practice creating and analyzing social media campaigns across different platforms.",

    "email marketing": "Learn email marketing fundamentals and create a sample email campaign.",

    "seo": "Learn SEO fundamentals and practice optimizing a website for relevant search terms.",

    "keyword research": "Learn keyword research techniques and practice identifying relevant keywords for a website.",

    "google analytics": "Learn Google Analytics fundamentals and practice analyzing website traffic and user behavior.",

    "excellent written and verbal communication": "Practice professional writing, presentations, and clear verbal communication through real-world projects.",
}

# Job Scoring

def score_job(

    job,

    skills,

    resume_embedding,

    job_embedding,

    candidate_experience,

    experience_penalty,

    candidate_background_embedding

):

    matching_skills = []

    # Targeted aliases for legitimate skill-name variations
    TARGETED_SKILL_ALIASES = {
        "python programming": ["python"],
        "python programming skills": ["python"],
        "python programming experience": ["python"],
        "large language models (llms)": ["llms", "llm"],
        "large language models": ["llms", "llm"],
        "large language model": ["llm", "llms"],
        "retrieval-augmented generation (rag)": ["rag"],
        "retrieval augmented generation": ["rag"],
        "ai agents": ["ai agents", "ai agent"],
        "embeddings and semantic search": [
            "embeddings",
            "semantic search"
        ],
        "knowledge of prompt engineering": [
             "prompt engineering"
        ],
    }

    # Related terminology that can legitimately represent a skill
    skill_context_aliases = {
        "api integration": [
            "openai api",
            "api integration",
            "rest api",
            "rest apis"
        ],

        "rest apis": [
            "openai api",
            "rest api",
            "rest apis",
            "api integration"
        ],

        "automation": [
            "ai automation",
            "automation workflows",
            "automated workflows"
        ],

        "error handling": [
            "error handling",
            "api-failure handling",
            "api failure handling",
            "api error handling",
            "failure handling",
            "exception handling",
            "fault handling",
            "failure recovery"
        ],

        "api testing": [
            "automated testing",
            "api testing"
        ]
    }

    project_text = ""

    for project in projects:
        project_text += " " + project.get("name", "")
        project_text += " " + " ".join(project.get("technologies", []))
        project_text += " " + project.get("description", "")
        project_text += " " + " ".join(project.get("responsibilities", []))
        project_text += " " + " ".join(project.get("outcomes", []))

    experience_text = ""

    for experience in profile_data["experience"]:
        experience_text += " " + experience.get("job_title", "")
        experience_text += " " + experience.get("company", "")

    candidate_skill_text = (
        " ".join(skills)
        + " "
        + project_text
        + " "
        + experience_text
    ).lower()

    for required_skill in job["required_skills"]:

        required_normalized = required_skill.strip().lower()

        aliases = SKILL_ALIASES.get(
            required_normalized,
            []
        )

        aliases = aliases + TARGETED_SKILL_ALIASES.get(
            required_normalized,
            []
        )

        # Normalize punctuation
        normalized_required = (
            required_normalized
            .replace("(", " ")
            .replace(")", " ")
            .replace("-", " ")
        )

        normalized_candidate_text = (
            candidate_skill_text
            .replace("(", " ")
            .replace(")", " ")
            .replace("-", " ")
        )

        # Check the required skill directly
        alias_match = bool(
            re.search(
                r"\b" + re.escape(normalized_required) + r"\b",
                normalized_candidate_text
            )
        )

        # Check targeted aliases
        if not alias_match:
            alias_match = any(
                re.search(
                    r"\b" + re.escape(
                        alias.lower()
                        .replace("(", " ")
                        .replace(")", " ")
                        .replace("-", " ")
                    ) + r"\b",
                    normalized_candidate_text
                )
                for alias in aliases
            )

        # Additional LLM naming variation
        if not alias_match:
            if required_normalized in [
                "large language models",
                "large language model",
                "large language models (llms)"
            ]:
                alias_match = (
                    re.search(
                        r"\bllms?\b",
                        normalized_candidate_text
                    )
                    is not None
                    or
                    re.search(
                        r"\blarge language models?\b",
                        normalized_candidate_text
                    )
                    is not None
                )

        # Automation variation
        if not alias_match:
            if required_normalized == "automation":
                alias_match = (
                    "automation" in normalized_candidate_text
                )

        if alias_match:
            if required_skill not in matching_skills:
                matching_skills.append(required_skill)
            continue

        # Check legitimate contextual skill representations
        context_aliases = skill_context_aliases.get(
            required_normalized,
            []
        )

        context_match = any(
            re.search(
                r"\b" + re.escape(alias.lower()) + r"\b",
                normalized_candidate_text
            )
            for alias in context_aliases
        )

        if context_match:
            if required_skill not in matching_skills:
                matching_skills.append(required_skill)
            continue

    # Calculate match percentage AFTER checking all required skills
    match_count = len(matching_skills)

    if job["required_skills"]:
        match_percentage = (
            match_count / len(job["required_skills"])
        ) * 100
    else:
        match_percentage = 0

    semantic_score = calculate_similarity(
        resume_embedding,
        job_embedding
    )

    job_responsibilities_text = ""

    for responsibility in job.get("responsibilities", []):
        job_responsibilities_text += "\n" + responsibility

    job_responsibilities_embedding = create_embedding(
        job_responsibilities_text
    )

    responsibility_similarity = calculate_similarity(
        candidate_background_embedding,
        job_responsibilities_embedding
    )

    responsibility_score = responsibility_similarity * 100

    exact_score = match_percentage / 100

    combined_score = (
        exact_score * 0.50
        + semantic_score * 0.30
        + responsibility_similarity * 0.20
)

    combined_percentage = combined_score * 100

    experience_warning = None

    experience_gap = max(
    job["experience_required"] - candidate_experience,
    0
)
    if experience_gap == 1:
        experience_label = "1 year"
    else:
        experience_label = f"{experience_gap} years"

    experience_penalty_applied = 0


    if experience_gap > 0:
        gap_penalty = min(experience_gap * 0.10, experience_penalty)
        experience_penalty_applied = gap_penalty
        combined_percentage *= (1 - gap_penalty)
        experience_warning = "Experience requirement not met."



    if combined_percentage >= 80:
        match_category = "Strong Match"
    elif combined_percentage >= 60:
        match_category = "Good Match"
    elif combined_percentage >= 40:
        match_category = "Partial Match"
    else:
        match_category = "Low Match"

    matching_skills_normalized = {
        skill.strip().lower()
        for skill in matching_skills
    }

    missing_skills = [
        skill
        for skill in job["required_skills"]
        if skill.strip().lower() not in matching_skills_normalized
    ]

    skill_priorities = {}

    responsibility_text = " ".join(
        job.get("responsibilities", [])
    ).lower()

    for skill in missing_skills:
        skill_lower = skill.lower()

        skill_variants = [skill_lower]

        for alias in SKILL_ALIASES.get(skill_lower, []):
            skill_variants.append(alias.lower())

        has_priority_match = any(
            re.search(
                r"\b" + re.escape(variant) + r"\b",
                responsibility_text
            )
            for variant in skill_variants
        )

        if has_priority_match:
            skill_priorities[skill] = "High"
        else:
            skill_priorities[skill] = "Medium"


    if match_category == "Strong Match" and experience_gap == 0:
        application_readiness = "High"

    elif match_category in ["Strong Match", "Good Match"]:
         if experience_gap == 0:
            application_readiness = "High"
         else:
            application_readiness = "Moderate"

    elif match_category == "Partial Match":
        application_readiness = "Moderate"

    else:
        application_readiness = "Low"


    if match_category == "Strong Match" and experience_gap == 0:

        application_advice = (
            "Apply confidently. Your skills and experience meet the "
            "requirements of this role."
        )

    elif match_category in ["Strong Match", "Good Match"]:

        if (
            experience_gap > 0
            and project_count > 0
            and responsibility_similarity >= PROJECT_RELEVANCE_THRESHOLD
            and missing_skills
       ):
            missing_skills_text = ", ".join(missing_skills)
            application_advice = (
                "Consider applying. Your project experience is relevant, "
                "but you should strengthen your professional experience "
                f"and address the following missing skill(s): {missing_skills_text}."
            )

        elif (
            experience_gap > 0
            and project_count > 0
            and responsibility_similarity >= PROJECT_RELEVANCE_THRESHOLD
       ):
            application_advice = (
                "Consider applying. Your skills and project experience "
                "are relevant, but you should strengthen your professional "
                "experience."
            )

        elif experience_gap > 0 and missing_skills:
            application_advice = (
                "Consider applying if the experience requirement is flexible. "
                "Your skills are relevant, but you have an experience gap "
                "and should address the missing skills."
           )

        elif experience_gap > 0:
            application_advice = (
                "Consider applying if the experience requirement is flexible. "
                "Your skills are relevant, but you have an experience gap."
           )

        else:
            application_advice = (
                "Consider applying. Your skills are a good match for this role."
            )

    elif match_category == "Partial Match":

        application_advice = (
            "Consider applying only if you can demonstrate transferable skills "
            "and address the identified gaps."
        )

    else:

        if experience_gap > 0 and missing_skills:
            application_advice = (
                "This role is currently a low match. Focus on building the missing "
                "skills and gaining the required professional experience before "
                "targeting similar roles."
           )

        elif experience_gap > 0:
            application_advice = (
                "This role is currently a low match. Focus on gaining the required "
                "professional experience before targeting similar roles."
           )

        else:
            application_advice = (
                "This role is currently a low match. Focus on building the missing "
                "skills before targeting similar roles."
            )

        


    if match_category == "Strong Match":
        overall_assessment = "Strong alignment with the job requirements."

    elif match_category == "Good Match":
        overall_assessment = "Good alignment with the job requirements."

    elif match_category == "Partial Match":
        overall_assessment = "Partial alignment with the job requirements."

    else:
        overall_assessment = "Limited alignment with the job requirements."


    if matching_skills:
        overall_assessment += (
            " Matching skills: "
            + ", ".join(matching_skills)
            + "."
       )
    else:
        overall_assessment += " No required skills matched."


    if missing_skills:
        overall_assessment += (
            " Skills to improve: "
            + ", ".join(missing_skills)
            + "."
        )
    else:
        overall_assessment += " No major skill gaps identified."


    if (
        project_count > 0
        and responsibility_similarity >= PROJECT_RELEVANCE_THRESHOLD
   ):
        project_label = "project" if project_count == 1 else "projects"
        overall_assessment += (
            f" The candidate has {project_count} relevant "
            f"{project_label}."
       )
    elif project_count > 0:
        overall_assessment += (
            f" The candidate has {project_count} project"
            f"{'s' if project_count != 1 else ''}, "
            "but the project responsibilities do not strongly align "
            "with the job responsibilities."
        )
    else:
        overall_assessment += " The candidate has no relevant projects listed."


    if experience_gap > 0:
        overall_assessment += (
            f" The candidate has an experience gap of "
            f"{experience_label}."
       )


    score_explanation = (
        f"Matched {len(matching_skills)} of "
        f"{len(job['required_skills'])} required skills. "
        f"Exact skill score: {exact_score * 100:.2f}% "
        f"(50% weight). "
        f"Semantic score: {semantic_score * 100:.2f}% "
        f"(30% weight). "
        f"Responsibility match: {responsibility_score:.2f}% "
        f"(20% weight)."
    )

    if missing_skills:
        score_explanation += (
            " Missing skills: "
            + ", ".join(missing_skills)
            + "."
        )

    if experience_gap > 0:
        score_explanation += (
            f" Experience gap: {experience_label}. "
            f"A {experience_penalty_applied * 100:.1f}% "
            "experience penalty was applied because the candidate "
            "does not meet the required experience."

        )

    return {
        "title": job["title"],
        "company": job["company"],
        "experience_required": job["experience_required"],
        "experience_gap": experience_gap,
        "experience_penalty_applied": round(
            experience_penalty_applied * 100, 2
        ),
        "exact_score": round(exact_score * 100, 2),
        "semantic_score": round(semantic_score * 100, 2),
        "responsibility_score": round(
            responsibility_score,
            2
        ),
        "responsibility_similarity": responsibility_similarity,
        "match_percentage": round(combined_percentage, 2),
        "match_category": match_category,
        "application_readiness": application_readiness,
        "application_advice": application_advice,
        "matching_skills": matching_skills,
        "strengths": matching_skills + (
            ["Relevant project experience"]
            if project_count > 0
            and responsibility_similarity >= PROJECT_RELEVANCE_THRESHOLD
            else []
     ),
        "missing_skills": missing_skills,
        "skill_priorities": skill_priorities,
        "experience_warning": experience_warning,
        "score_explanation": score_explanation,
        "overall_assessment": overall_assessment 
    }

job_profile = extract_job_profile(job_posting_text)

job_data = json.loads(job_profile)

if not job_data.get("company"):
    job_data["company"] = "User Provided Job"

job_embedding = create_job_embedding(job_data)

# Job Analysis

job_result = score_job(

    job_data,

    skills,

    resume_embedding,

    job_embedding,

    candidate_experience,

    experience_penalty,

    candidate_background_embedding

)

# Candidate Profile Output

if skills:
    core_ai_skills = []
    development_skills = []
    ai_reliability_skills = []
    tools_and_platforms = []
    other_skills = []

    for skill in skills:
        skill_lower = skill.lower()

        if any(
            keyword in skill_lower
            for keyword in [
                "llm",
                "rag",
                "embedding",
                "semantic search",
                "ai agent",
                "prompt engineering",
                "automation"
            ]
        ):
            core_ai_skills.append(skill)

        elif any(
            keyword in skill_lower
            for keyword in [
                "python",
                "git",
                "github",
                "terminal",
                "vs code"
            ]
        ):
            development_skills.append(skill)

        elif any(
            keyword in skill_lower
            for keyword in [
                "guardrail",
                "red teaming",
                "evaluation",
                "testing",
                "logging",
                "failure handling",
                "escalation"
            ]
        ):
            ai_reliability_skills.append(skill)

        elif any(
            keyword in skill_lower
            for keyword in [
                "openai",
                "api",
                "json"
            ]
        ):
            tools_and_platforms.append(skill)

        else:
            other_skills.append(skill)

    if core_ai_skills:
        print("\nCore AI Skills:")
        for skill in core_ai_skills:
            print("-", skill)

    if development_skills:
        print("\nDevelopment:")
        for skill in development_skills:
            print("-", skill)

    if ai_reliability_skills:
        print("\nAI Reliability & Evaluation:")
        for skill in ai_reliability_skills:
            print("-", skill)

    if tools_and_platforms:
        print("\nTools & Platforms:")
        for skill in tools_and_platforms:
            print("-", skill)

    if other_skills:
        print("\nOther Skills:")
        for skill in other_skills:
            print("-", skill)

else:
    print("None")

print("\nProjects:")

if projects:
    for project in projects:
        project_name = project.get(
            "name",
            "Project name not provided"
        )

        print("-", project_name)

        description = project.get("description", "")
        if description:
            print("  Description:", description)

        project_skills = project.get("skills", [])

        if project_skills:
            valid_project_skills = [
                skill
                for skill in project_skills
                if skill in skills
            ]

            if valid_project_skills:
                print(
                    "  Skills:",
                    ", ".join(valid_project_skills)
                )
else:
    print("None")

print("\nEducation:")
if education:
    

    for item in education:
        if isinstance(item, dict):
            institution = item.get("institution", item.get("school", ""))
            degree = item.get("degree", "")

            if institution and degree:
                print("-", degree, "at", institution)
            elif institution:
                print("-", institution)
            elif degree:
                print("-", degree)
        else:
            print("-", item)
else:
    print("None")

print("\nProfessional Experience:")
if experience:
    for item in experience:
        job_title = item.get("job_title", "")
        company = item.get("company", "")

        if job_title and company:
            print("-", job_title, "at", company)
        elif job_title:
            print("-", job_title)
        elif company:
            print("-", company)
else:
    print("None")

    # Candidate Summary

print("\nCandidate Summary:")

summary_parts = []

if skills:
    summary_parts.append(
        "The candidate demonstrates a technical skill set that includes "
        + ", ".join(skills[:6])
        + "."
    )

if projects:
    project_names = [
        project.get("name", "Unnamed project")
        for project in projects
    ]

    summary_parts.append(
        f"The candidate has {project_count} listed project"
        + ("." if project_count == 1 else "s.")
    )

    summary_parts.append(
        "Project work includes "
        + ", ".join(project_names)
        + "."
    )
else:
    summary_parts.append(
        "No projects were identified in the resume."
    )

if experience:
    experience_titles = [
        item.get("job_title", "unspecified role")
        for item in experience
    ]

    summary_parts.append(
        "Professional experience includes "
        + ", ".join(experience_titles)
        + "."
    )
else:
    summary_parts.append(
        "No professional experience was identified in the resume."
    )

print(" ".join(summary_parts))

# Candidate Strengths Summary

print("\nCandidate Strengths:")

if skills:
    print("- Strong technical foundation across:")
    for skill in skills[:6]:
        print("  -", skill)
else:
    print("- No specific skills were identified.")

if projects:
    print(
        "- Demonstrated practical project work through:",
        ", ".join(
            project.get("name", "Unnamed project")
            for project in projects
        )
    )
else:
    print("- No projects were identified.")

if experience:
    print("- Professional experience is present.")
else:
    print("- No professional experience was identified.")


print("\nCandidate Experience Summary:")

if experience:
    total_experience = sum(
        float(item.get("years", 0) or 0)
        for item in experience
    )

    print(
        f"- Professional experience identified: "
        f"{total_experience:.1f} years."
    )

    for item in experience:
        job_title = item.get("job_title", "")
        company = item.get("company", "")

        if job_title and company:
            print("-", job_title, "at", company)
        elif job_title:
            print("-", job_title)
        elif company:
            print("-", company)
else:
    print("- No professional employment experience identified.")

if projects:
    print(
        f"- Project experience identified: "
        f"{project_count} listed project"
        + ("." if project_count == 1 else "s.")
    )
else:
    print("- No project experience identified.")


# Job Analysis Output

print("\n" + "=" * 40)
print("           JOB ANALYSIS")
print("=" * 40)

print("\nJob Title:", job_result["title"])
print("Company:", job_result["company"])

print("\nMatch:", job_result["match_percentage"], "%")
print("Category:", job_result["match_category"])

print(
    "Application Readiness:",
    job_result["application_readiness"]
)

print("\nScore Breakdown:")
print(
    "- Exact Skills:",
    job_result["exact_score"],
    "% (50% weight)"
)
print(
    "- Semantic Similarity:",
    job_result["semantic_score"],
    "% (30% weight)"
)
print(
    "- Responsibility Match:",
    job_result["responsibility_score"],
    "% (20% weight)"
)
print(
    "- Experience Penalty:",
    job_result["experience_penalty_applied"],
    "%"
)

# Skill Match Output

print("\nMatching Skills:")
if job_result["matching_skills"]:
    for skill in job_result["matching_skills"]:
        print("-", skill)
else:
    print("None")

print("\nMissing Skills:")

if job_result["missing_skills"]:

    for skill in job_result["missing_skills"]:

        priority = job_result["skill_priorities"].get(
            skill,
            "Medium"
        )

        print(
            "-",
            skill,
            "|",
            priority,
            "Priority"
        )

else:

    print("None")

# Improvement Recommendations Output

print("\nHow to improve:")

if job_result["missing_skills"]:

    prioritized_skills = sorted(
        job_result["missing_skills"],
        key=lambda skill: (
            job_result["skill_priorities"].get(skill, "Medium") != "High"
        )
    )

    for skill in prioritized_skills:
        skill_key = skill.lower()
        priority = job_result["skill_priorities"].get(skill, "Medium")

        if skill_key in SKILL_RECOMMENDATIONS:
            print(
                "-",
                skill,
                f"({priority} Priority):",
                SKILL_RECOMMENDATIONS[skill_key]
            )
        else:
            print(
                "-",
                skill,
                f"({priority} Priority):",
                "Learn the fundamentals of this skill and build a practical project to demonstrate it."
            )

else:

    print("None — all required skills are matched.")

# Strengths Output

print("\nStrengths:")

if job_result["strengths"]:
    for strength in job_result["strengths"]:
        print("-", strength)
else:
    print("None")


# Why You're a Match

print("\n" + "=" * 40)
print("          WHY YOU'RE A MATCH")
print("=" * 40)

print("\nMatching Skills:")

if job_result["matching_skills"]:
    for skill in job_result["matching_skills"]:
        print("-", skill)
else:
    print("None")


print("\nRelevant Experience:")

if project_count > 0:
    print("- Project experience identified:")

    for project in projects:
        print(
            "-",
            project.get("name", "Unnamed project")
        )
else:
    print("- No project experience identified.")

print("\nKey Gaps:")

if job_result["missing_skills"]:
    for skill in job_result["missing_skills"]:
        print("-", skill, "is required but is not currently demonstrated.")
else:
    print("No required skill gaps identified.")


print("\nExperience Gap:")

if job_result["experience_gap"] > 0:
    required_years_label = (
        "1 year"
        if job_result["experience_required"] == 1
        else f"{job_result['experience_required']} years"
    )

    candidate_years_label = (
        "1 year"
        if candidate_experience == 1
        else f"{candidate_experience} years"
    )

    print(
        "- The role requires",
        required_years_label,
        "of professional experience."
    )
    print(
        "- The candidate has",
        candidate_years_label,
        "of professional experience."
    )
else:
    print("- The experience requirement is met.")


# Candidate Experience and Projects Output

print("\nProfessional Experience:", candidate_experience, "years")
print("Projects Listed:", project_count)

if projects:
    print("Project Names:")
    for project in projects:
        print("-", project.get("name", "Project name not provided"))

# Experience Requirement Output

print(
    "\nExperience Required:",
    "1 year" if job_result["experience_required"] == 1
    else f"{job_result['experience_required']} years"
)

print(
    "Experience Gap:",
    "1 year" if job_result["experience_gap"] == 1
    else f"{job_result['experience_gap']} years"
)

print(
    "Experience Penalty Applied:",
    job_result["experience_penalty_applied"],
    "%"
)

# Candidate Development Plan Output

print("\n" + "=" * 40)
print("       CANDIDATE DEVELOPMENT PLAN")
print("=" * 40)

if job_result["missing_skills"]:

    print("\nTop Skills to Develop:")

    prioritized_skills = sorted(
        job_result["missing_skills"],
        key=lambda skill: (
            job_result["skill_priorities"].get(skill, "Medium") != "High"
        )
   )

    for index, skill in enumerate(prioritized_skills[:3], start=1):
        priority = job_result["skill_priorities"].get(skill, "Medium")
        print(index, ".", skill, "-", priority, "Priority")

else:
    print("\nTop Skills to Develop:")
    print("None — all required skills are matched.")

if job_result["experience_gap"] > 0:

    print("\nExperience to Gain:")

    if job_result["experience_gap"] == 1:
        print("- 1 year of relevant professional experience.")
    else:
        print(
            "-",
            job_result["experience_gap"],
            "years of relevant professional experience."
        )

else:
    print("\nExperience to Gain:")
    print("None — experience requirement is met.")

# Project Recommendation Output

print("\nProject Recommendation:")

if job_result["missing_skills"]:

    recommended_skills = job_result["missing_skills"][:4]

    if (
        project_count > 0
        and job_result["responsibility_similarity"]
        >= PROJECT_RELEVANCE_THRESHOLD
    ):
        print(
            "- Strengthen your existing project(s) by demonstrating:",
            ", ".join(recommended_skills) + "."
        )

    else:
        print(
            "- Build a practical project that demonstrates:",
            ", ".join(recommended_skills) + "."
        )

else:

    print(
        "- No additional project is required for the matched skills."
    )

# Application Advice Output

print("\nApplication Advice:")
print("-", job_result["application_advice"])

# Final Recommendation Output

print("\nFINAL RECOMMENDATION")
print("--------------------")

if job_result["match_category"] == "Strong Match" and job_result["experience_gap"] == 0:
    print("Apply confidently")

elif job_result["match_category"] == "Good Match" and job_result["experience_gap"] == 0:
    print("Apply")

elif job_result["match_category"] == "Good Match" and job_result["experience_gap"] > 0:
    print("Apply with caution")

elif job_result["match_category"] == "Partial Match":
    print("Consider applying after addressing key gaps")

else:
    print("Build skills and experience before applying")

print("\nJob Match Result:")
print("Match:", job_result["match_percentage"], "%")
print("Category:", job_result["match_category"])
print("Matching skills:", job_result["matching_skills"])
print("Missing skills:", job_result["missing_skills"])
print("Exact skill score:", job_result["exact_score"], "%")
print("Semantic score:", job_result["semantic_score"], "%")
print(
    "Responsibility score:",
    job_result["responsibility_score"],
    "%"
)
print("Why this score:", job_result["score_explanation"])
if job_result["experience_required"] == 1:
    experience_required_label = "1 year"
else:
    experience_required_label = f"{job_result['experience_required']} years"

if job_result["experience_gap"] == 1:
    experience_gap_label = "1 year"
else:
    experience_gap_label = f"{job_result['experience_gap']} years"

print("Experience requirement:", experience_required_label)

print("Experience gap:", experience_gap_label)
print(
    "Experience penalty applied:",
    job_result["experience_penalty_applied"],
    "%"
)


job_results = []

job_embeddings = {}

for job in jobs:
    job_embeddings[job["title"]] = create_job_embedding(job)

for job in jobs:

    result = score_job(

        job,

        skills,

        resume_embedding,

        job_embeddings[job["title"]],

        candidate_experience,

        experience_penalty,

        candidate_background_embedding

    )

    job_results.append(result)


job_results.sort(
    key=lambda job: job["match_percentage"],
    reverse=True
)

# Reference Job Rankings Output

print("\n" + "=" * 40)
print("        REFERENCE JOB RANKINGS")
print("=" * 40)

for index, job in enumerate(job_results, start=1):

    print(
        index,
        ".",
        job["title"],
        "-",
        job["match_percentage"],
        "%",
        "|",
        job["match_category"]
    )
