"""
Full pipeline demo with a realistic multi-document corpus.

Simulates a company knowledge base with multiple documents
and runs the complete RAG pipeline end-to-end.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.pipeline import RAGPipeline
from src.eval import EvaluationHarness, LabeledExample


def main() -> None:
    print("=" * 80)
    print("Demo: Full Pipeline with Realistic Corpus")
    print("=" * 80)

    # Multi-document corpus simulating a company knowledge base
    documents = {
        "engineering_handbook.md": """
# Engineering Handbook

## Code Review Process

All code changes require at least one approval from a senior engineer.
Pull requests must include tests and documentation updates.
The CI pipeline runs automated tests, linting, and security scans.
Deployments to production require approval from the tech lead.

## Onboarding

New engineers receive a laptop, access to all internal tools, and a mentor.
The first week includes setup of development environment and reading key documentation.
By the end of the first month, engineers are expected to complete their first production deployment.
""",
        "security_policy.md": """
# Security Policy

## Access Control

All employees must use two-factor authentication for company systems.
Passwords must be at least 16 characters and rotated every 90 days.
Access to production systems requires approval from the security team.

## Incident Response

Security incidents must be reported to security@company.com within 1 hour.
The incident response team will investigate and provide remediation steps.
All incidents are documented and reviewed in the monthly security meeting.
""",
        "pto_policy.md": """
# PTO Policy

## Eligibility

Full-time employees are eligible for PTO starting on their first day.
New hires accrue PTO at a rate of 1.25 days per month during their first year.
After one year, the accrual rate increases to 1.67 days per month.

## Usage

PTO requests must be submitted at least 2 weeks in advance.
Requests during peak business periods (November, December) require additional approval.
Unused PTO can be carried over up to a maximum of 10 days.
""",
        "api_documentation.md": """
# API Documentation

## Authentication

All API requests require a Bearer token in the Authorization header.
Tokens expire after 24 hours and must be refreshed using the /auth/refresh endpoint.
Rate limits are 1000 requests per minute per API key.

## Endpoints

GET /api/v1/users - List all users
POST /api/v1/users - Create a new user
GET /api/v1/users/:id - Get user by ID
PUT /api/v1/users/:id - Update user
DELETE /api/v1/users/:id - Delete user

All responses are JSON. Errors include a message and error code.
""",
        "deployment_runbook.md": """
# Deployment Runbook

## Pre-Deployment Checklist

1. Ensure all tests pass in CI
2. Verify database migrations are up to date
3. Confirm monitoring dashboards are healthy
4. Notify the team in #deployments channel

## Deployment Steps

1. Merge PR to main branch
2. CI automatically builds and pushes Docker image
3. Update Kubernetes deployment with new image tag
4. Rolling update with zero downtime
5. Verify health checks pass
6. Monitor error rates for 30 minutes

## Rollback

If issues are detected, rollback to the previous stable version using:
kubectl rollout undo deployment/app
""",
    }

    # Combine all documents into a single corpus
    full_text = "\n\n".join(documents.values())

    # Initialize and index pipeline
    pipeline = RAGPipeline(persist_dir="data/index")
    pipeline.index(full_text, source="company_kb")
    print(f"Indexed {len(full_text.split())} words from {len(documents)} documents")
    print(f"Pipeline ready: {pipeline.indexed}\n")

    # Query examples
    queries = [
        "What is the code review process?",
        "How do I request PTO?",
        "What are the API authentication requirements?",
        "How do I deploy to production?",
        "What happens if there is a security incident?",
    ]

    print("=" * 80)
    print("Querying the Pipeline")
    print("=" * 80)

    for query in queries:
        print(f"\nQ: {query}")
        response = pipeline.query(query, top_k=3)
        print(f"A: {response.answer[:200]}...")
        print(f"   Sources: {[s.source_file for s in response.sources]}")
        print(f"   Latency: {response.chunks_retrieved} chunks retrieved, {response.chunks_used} used")

    # Evaluation
    print("\n" + "=" * 80)
    print("Evaluation")
    print("=" * 80)

    labeled_examples = [
        LabeledExample(
            question="What is the code review process?",
            answer="Code changes require approval from a senior engineer, tests, and documentation.",
            relevant_chunk_ids=[0],
            required_facts=["approval", "senior engineer", "tests"],
        ),
        LabeledExample(
            question="How do I request PTO?",
            answer="Submit PTO requests at least 2 weeks in advance.",
            relevant_chunk_ids=[2],
            required_facts=["2 weeks", "PTO request"],
        ),
        LabeledExample(
            question="What are the API authentication requirements?",
            answer="API requires Bearer token in Authorization header, expires after 24 hours.",
            relevant_chunk_ids=[3],
            required_facts=["Bearer token", "24 hours"],
        ),
    ]

    harness = EvaluationHarness(pipeline=pipeline, labeled_examples=labeled_examples, top_k=3)
    report = harness.evaluate()

    print(f"\nRetrieval Metrics:")
    print(f"  Precision@3: {report.retrieval.precision_at_k:.2f}")
    print(f"  Recall@3:    {report.retrieval.recall_at_k:.2f}")
    print(f"  MRR:         {report.retrieval.mrr:.2f}")
    print(f"\nGeneration Metrics:")
    print(f"  Faithfulness: {report.generation.faithfulness:.2f}")
    print(f"  Groundedness: {report.generation.groundedness:.2f}")

    harness.save_report(report, "data/evaluation_report.json")
    print(f"\nReport saved to data/evaluation_report.json")


if __name__ == "__main__":
    main()
