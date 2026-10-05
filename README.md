# AI Resume Analyzer

An AI-powered resume analysis and job matching application built with Python.

The application extracts structured candidate information from PDF resumes, analyzes job postings, and evaluates candidate-job compatibility using skill matching, semantic similarity, responsibility matching, and experience analysis.

## Overview

AI Resume Analyzer is a Python-based application that analyzes a candidate's resume against a job posting and evaluates how well the candidate matches the role.

The system extracts candidate skills, projects, education, and experience from a PDF resume, then compares that profile against the requirements and responsibilities of a job.

It combines exact skill matching, semantic similarity, responsibility matching, and experience-gap analysis to produce an overall job-match score and application recommendation.

The project was built to demonstrate practical skills in Python, large language models, embeddings, semantic search, RAG, AI evaluation, and AI-assisted decision support.

## Features

- PDF resume extraction
- Candidate profile extraction
- Job description analysis
- Required skill extraction
- Skill matching and alias detection
- Semantic similarity scoring
- Responsibility matching
- Experience-gap analysis
- Experience penalty calculation
- Missing-skill identification
- Candidate strengths analysis
- Development recommendations
- Application readiness assessment
- Final job recommendation
- Reference job ranking

## How It Works

The application follows this workflow:

1. Extracts information from the candidate's resume.
2. Builds a structured candidate profile.
3. Accepts a job posting from the user.
4. Extracts the job title, required skills, experience requirements, and responsibilities.
5. Compares the job requirements against the candidate profile.
6. Calculates:
   - Exact skill score
   - Semantic similarity score
   - Responsibility match score
   - Experience gap
   - Experience penalty
7. Produces an overall job match percentage.
8. Identifies matching and missing skills.
9. Provides development recommendations.
10. Generates an application recommendation.

## Scoring System

The overall job-match score is calculated using three weighted components:

- Exact Skills: 50%
- Semantic Similarity: 30%
- Responsibility Match: 20%

Experience requirements are evaluated separately through experience-gap analysis and experience-penalty logic.

### Score Components

**Exact Skill Score (50%)**  
Measures how many of the required job skills are directly matched by the candidate's demonstrated skills and recognized skill aliases.

**Semantic Similarity Score (30%)**  
Measures the semantic similarity between the candidate's profile and the job requirements.

**Responsibility Match Score (20%)**  
Measures how closely the candidate's demonstrated project and experience responsibilities align with the responsibilities described in the job posting.

**Experience Analysis**  
The system compares the required experience with the candidate's professional experience and calculates an experience gap and, when applicable, an experience penalty.

The final result combines these scoring components to produce a job-match percentage and match category.

## Example

For an AI Support Engineer position requiring Python, LLMs, RAG, embeddings, semantic search, API integration, error handling, Git, GitHub, and automated testing, the system can identify matching skills and evaluate the candidate's project experience.

### Example Result

- Match: 82.48%
- Category: Strong Match
- Exact Skills: 100%
- Semantic Similarity: 64.54%
- Responsibility Match: 65.57%
- Experience Penalty: 0%
- Recommendation: Apply confidently

## Project

### Nova AI Customer Support System

Nova AI Customer Support System is a semantic Retrieval-Augmented Generation (RAG) customer-support assistant built to ground AI responses in a structured customer-support knowledge base.

The system is designed to safely handle unsupported requests and demonstrates practical implementation of AI application reliability and evaluation.

The project demonstrates experience with:

- Python
- Large Language Models (LLMs)
- RAG
- Embeddings
- Semantic Search
- API Integration
- Error Handling
- Automated Testing
- AI Evaluation
- Git and GitHub

## Technologies

- Python
- Large Language Models (LLMs)
- RAG
- Embeddings
- Semantic Search
- AI Agents
- AI Automation
- OpenAI API
- Git
- GitHub
- JSON

## Education

Ibadan Polytechnic

## Current Experience Level

The candidate currently has no professional employment experience.

The system is designed to recognize personal and academic projects when evaluating entry-level opportunities.

## Purpose

This project was built to demonstrate practical skills in:

- AI application development
- Resume analysis
- Natural language processing
- Semantic similarity
- AI-assisted decision support
- Job matching
- Candidate skill analysis

## Status

Completed core functionality and testing.

The application successfully analyzes job postings and produces structured candidate-job matching results.