"""
AI Auditor Agent Module (agent.py)
Implements an active compliance auditor agent using a ReAct (Reasoning + Action) loop.
Equips the LLM with tools to query Qdrant regulations, fetch active fund metrics from Project 2,
and register compliance warnings.
"""

import json
import logging
import requests
from typing import List, Dict, Any, Tuple

from config import HF_MODEL_ID, HF_API_TOKEN
from vector_store import RegulatoryVectorStore
from reranker import LocalReranker

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("AuditorAgent")


class ComplianceTools:
    """Defines the active tools available to the Auditor Agent."""

    def __init__(self, vector_store: RegulatoryVectorStore, reranker: LocalReranker):
        self.vector_store = vector_store
        self.reranker = reranker

    def query_regulations(self, query: str) -> str:
        """
        Queries the Qdrant database for specific regulatory article clauses.
        Use this when you need to check what the legal limit or rule is.
        """
        logger.info(f"[Tool: query_regulations] Running search for: {query}")
        results = self.vector_store.search_semantic(query, top_k=5)
        reranked = self.reranker.rerank_documents(query, results, top_n=2)
        
        if not reranked:
            return "No matching regulatory clauses found."
            
        context_summary = "\n".join([
            f"- [{r['metadata']['hierarchy_path']}] (Relevance: {r['rerank_score']:.3f}): {r['text']}"
            for r in reranked
        ])
        return context_summary

    def get_fund_metrics(self, fund_id: str) -> str:
        """
        Fetches the current cash buffers, AUM, and fees of a simulated fund.
        Use this when you need to audit a fund's actual parameters.
        """
        logger.info(f"[Tool: get_fund_metrics] Querying databases for fund ID: {fund_id}")
        
        # In a production system, this queries the Project 2 database API.
        # We simulate the response metrics:
        # Fund is in violation: Cash ratio is 0.5% (target is 2%)
        mock_fund_database = {
            "LU123456789": {
                "Fund_Name": "Lux Growth Equities Fund",
                "AUM": 105000000.0,
                "Cash_Buffer": 525000.0,  # 0.5% Cash Buffer (Violation!)
                "Cash_Ratio": 0.005,
                "Management_Fee_Annual": 0.015
            },
            "LU987654321": {
                "Fund_Name": "Lux Safe Cash Reserve",
                "AUM": 200000000.0,
                "Cash_Buffer": 4400000.0,  # 2.2% Cash Buffer (Compliant)
                "Cash_Ratio": 0.022,
                "Management_Fee_Annual": 0.008
            }
        }
        
        fund_data = mock_fund_database.get(fund_id.upper())
        if not fund_data:
            return f"Error: Fund ID {fund_id} not found in the portfolio registry."
            
        return json.dumps(fund_data, indent=2)

    def submit_compliance_alert(self, report: str) -> str:
        """
        Registers an official operational violation warning in the compliance logs.
        Use this when a fund parameter breaches a regulatory limit.
        """
        logger.info(f"[Tool: submit_compliance_alert] Alert registered: {report}")
        return f"SUCCESS: Compliance warning logged. Alert details: {report}"


class AuditorAgent:
    """
    Orchestrates the ReAct loop, allowing the LLM to write thoughts,
    call tools, observe outcomes, and formulate final audit verdicts.
    """

    def __init__(self, tools: ComplianceTools):
        self.tools = tools
        self.max_loops = 4

    def run_audit(self, query_task: str) -> Dict[str, Any]:
        """
        Runs the reasoning loop to solve the audit query.
        """
        logger.info(f"Starting auditor agent run for task: {query_task}")
        
        # Build prompt showing the LLM the tools it can use and the format it must follow
        agent_system_prompt = (
            "You are an active AI Regulatory Auditor for Luxembourg funds.\n"
            "You solve compliance tasks by running a loop of Thought, Action, Observation.\n\n"
            "You have access to the following tools:\n"
            "1. Tool: query_regulations\n"
            "   Description: Queries regulatory PDFs for limits/rules. Input must be a query string.\n"
            "2. Tool: get_fund_metrics\n"
            "   Description: Fetches a fund's actual AUM/Cash metrics. Input must be a Fund ID (e.g. LU123456789).\n"
            "3. Tool: submit_compliance_alert\n"
            "   Description: Registers a violation in the logs. Input is the warning details string.\n\n"
            "To call a tool, output exactly this JSON format:\n"
            "{\n"
            "  \"Action\": \"tool_name\",\n"
            "  \"Input\": \"tool_input_value\"\n"
            "}\n\n"
            "If you have solved the task and reached the final audit result, output exactly this JSON format:\n"
            "{\n"
            "  \"Final_Answer\": \"your detailed compliance audit verdict\"\n"
            "}\n\n"
            "Let's begin. Task: " + query_task + "\n"
        )

        history = agent_system_prompt
        steps_taken = []

        for loop_idx in range(self.max_loops):
            logger.info(f"Agent Loop {loop_idx+1}/{self.max_loops}...")
            
            # Query the LLM
            llm_output = self._query_hf(history)
            logger.info(f"Agent Thought/Action: {llm_output}")
            
            try:
                # Parse JSON action
                action_data = self._extract_json(llm_output)
                
                if "Final_Answer" in action_data:
                    # Final answer reached
                    return {
                        "status": "success",
                        "verdict": action_data["Final_Answer"],
                        "steps": steps_taken
                    }
                
                # Execute action
                tool_name = action_data.get("Action")
                tool_input = action_data.get("Input")
                
                observation = ""
                if tool_name == "query_regulations":
                    observation = self.tools.query_regulations(tool_input)
                elif tool_name == "get_fund_metrics":
                    observation = self.tools.get_fund_metrics(tool_input)
                elif tool_name == "submit_compliance_alert":
                    observation = self.tools.submit_compliance_alert(tool_input)
                else:
                    observation = f"Error: Tool '{tool_name}' does not exist."

                logger.info(f"Agent Observation: {observation}")
                
                # Update history for next iteration
                history += f"\nOutput:\n{json.dumps(action_data)}\nObservation:\n{observation}\n"
                steps_taken.append({
                    "action": tool_name,
                    "input": tool_input,
                    "observation": observation
                })

            except Exception as parse_err:
                logger.error(f"Failed to parse agent action: {parse_err}")
                # If LLM failed to write JSON, we force it to output a final answer
                history += f"\nError: Your output must be a valid JSON containing either 'Action'/'Input' or 'Final_Answer'. Correct your format."
        
        return {
            "status": "timeout",
            "verdict": "Agent failed to converge within the limit.",
            "steps": steps_taken
        }

    def _extract_json(self, text: str) -> Dict[str, Any]:
        """Extracts JSON block from LLM output."""
        # Find JSON boundaries
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1:
            raise ValueError("No JSON block found in output.")
        json_str = text[start:end+1]
        return json.loads(json_str)

    def _query_hf(self, prompt: str) -> str:
        """Helper to query Hugging Face API."""
        if not HF_API_TOKEN:
            # Local fallback simulation if API token is missing
            if "get_fund_metrics" in prompt and "LU123456789" in prompt and "Observation" not in prompt:
                return '{"Action": "get_fund_metrics", "Input": "LU123456789"}'
            elif "query_regulations" in prompt and "Observation" not in prompt:
                return '{"Action": "query_regulations", "Input": "What is the cash limit for investment funds?"}'
            elif "LU123456789" in prompt and "Observation:" in prompt:
                return '{"Action": "submit_compliance_alert", "Input": "LU123456789 violates the 2% cash buffer rule with an active ratio of 0.5%."}'
            else:
                return '{"Final_Answer": "Fund LU123456789 holds 0.5% cash, violating the 2% regulatory limit established in CSSF Circular 18-698. Compliance alert has been registered."}'

        api_url = f"https://api-inference.huggingface.co/models/{HF_MODEL_ID}"
        headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}
        
        try:
            response = requests.post(
                api_url,
                headers=headers,
                json={
                    "inputs": prompt,
                    "parameters": {"max_new_tokens": 256, "temperature": 0.1, "return_full_text": False}
                },
                timeout=15
            )
            if response.status_code == 200:
                res_json = response.json()
                if isinstance(res_json, list) and len(res_json) > 0:
                    return res_json[0].get("generated_text", "")
            return '{"Final_Answer": "API limits hit, fallback resolved."}'
        except Exception:
            return '{"Final_Answer": "Connection timeout, fallback resolved."}'


if __name__ == "__main__":
    # Test stub
    from vector_store import RegulatoryVectorStore
    
    store = RegulatoryVectorStore()
    rerank = LocalReranker()
    tools = ComplianceTools(store, rerank)
    agent = AuditorAgent(tools)
    
    # Run mock audit
    res = agent.run_audit("Audit fund LU123456789 and check if it complies with cash buffer rules.")
    print("\n" + "="*50)
    print("FINAL AGENT VERDICT:")
    print(res["verdict"])
    print("="*50)
