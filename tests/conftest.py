import pytest


@pytest.fixture
def sample_space():
    return {
        "version": 2,
        "data_sources": {
            "tables": [
                {
                    "identifier": "sandbox.jg_banking_genie.account",
                    "column_configs": [{"column_name": "status"}],
                },
                {"identifier": "sandbox.jg_banking_genie.client_group"},
                {
                    "identifier": "sandbox.jg_banking_genie.client_party",
                    "column_configs": [
                        {"column_name": "party_name"},
                        {"column_name": "known_as"},
                    ],
                },
            ]
        },
        "instructions": {
            "text_instructions": [
                {"id": "30000000000000000000000000000001", "content": ["Identify clients with find_client."]}
            ],
            "join_specs": [
                {
                    "id": "60000000000000000000000000000001",
                    "left": {"identifier": "sandbox.jg_banking_genie.account"},
                    "right": {"identifier": "sandbox.jg_banking_genie.client_group"},
                    "sql": [
                        "account.group_id = client_group.group_id",
                        "--rt=FROM_RELATIONSHIP_TYPE_MANY_TO_ONE--",
                    ],
                    "instruction": ["Roll accounts up to the group."],
                }
            ],
            "sql_snippets": {"filters": [], "measures": []},
            "example_question_sqls": [],
        },
        "benchmarks": {
            "questions": [
                {
                    "id": "50000000000000000000000000000001",
                    "question": ["How many open accounts?"],
                    "answer": [
                        {
                            "format": "SQL",
                            "content": [
                                "SELECT COUNT(*) AS open_account_count\n",
                                "FROM sandbox.jg_banking_genie.account\n",
                                "WHERE status = 'Open'\n",
                            ],
                        }
                    ],
                }
            ]
        },
    }
