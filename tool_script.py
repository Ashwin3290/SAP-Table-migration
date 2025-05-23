#!/usr/bin/env python3
"""
Comprehensive Test Script for DMTool SQL System
Tests all transformation prompts from the requirements document
"""

import os
import sys
import time
import json
import logging
import traceback
import pandas as pd
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional

# Add current directory to path to import DMToolSQL
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from dmtool import DMToolSQL
except ImportError:
    print("Error: Could not import DMToolSQL. Please ensure the module is available.")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('dmtool_test_results.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class DMToolTester:
    """Comprehensive tester for DMTool SQL transformations"""
    
    def __init__(self):
        """Initialize the tester"""
        self.sql_tool = DMToolSQL()
        self.session_id = None
        self.test_results = []
        self.object_id = 41
        self.project_id = 24
        self.start_time = None
        self.total_tests = 0
        self.passed_tests = 0
        self.failed_tests = 0
        
    def define_test_cases(self) -> List[Dict[str, Any]]:
        """Define all test cases from the requirements document"""
        test_cases = [
            # Material Basic Segment Tests (577)
            {
                "id": "MB_001",
                "name": "Basic Material Number Extraction",
                "segment_id": 577,
                "query": "Bring Material Number with Material Type = ROH from MARA Table",
                "expected_type": "Simple Transformation",
                "description": "Extract material numbers with specific material type"
            },
            {
                "id": "MB_002", 
                "name": "Complex Unit of Measure Logic",
                "segment_id": 577,
                "query": """Check Materials which you have got from Transformation rule 1 In MARA_500 table and
IF matching Entries found, then bring Unit of Measure field from MARA_500 table to the Target Table
ELSE, If no entries found in MARA_500, then check ROH Material (found in Transformation 2) in MARA_700 Table and bring the Unit of Measure
ELSE, If no entries found in MARA_700, then bring the Unit of measure from MARA table""",
                "expected_type": "Conditional Transformation",
                "description": "Complex conditional logic for unit of measure with multiple fallbacks"
            },
            {
                "id": "MB_003",
                "name": "Complex Material Type Logic", 
                "segment_id": 577,
                "query": """Check ROH Material In MARA_500 table and
IF matching Entries found, then bring Material Type field from MARA_500 table to the Target Table
ELSE, If no entries found in MARA_500, then check ROH Material (found in Transformation 2) in MARA_700 Table and bring the Material Type
ELSE, bring the Material Type from MARA table""",
                "expected_type": "Conditional Transformation",
                "description": "Complex conditional logic for material type"
            },
            {
                "id": "MB_004",
                "name": "Material Group Mapping",
                "segment_id": 577,
                "query": """If Material Group in (L002, ZMGRP1, 202R) then 'GENAI01'
ELSE IF Material Group in ('01', 'L001', 'ZSUK1RM1') then 'GENAI02'  
ELSE IF Material Group in ('CH001', '02') then 'GenAI03'
ELSE 'GenAINo' (MARA Table)""",
                "expected_type": "Conditional Mapping",
                "description": "Map material groups to specific codes"
            },
            {
                "id": "MB_005",
                "name": "Industry Sector Null Handling",
                "segment_id": 577,
                "query": """IF Industry sector is null (from MARA) then try to bring from MARA_500
even if MARA_500 has null value for Industry sector, then hardcode as 'M'""",
                "expected_type": "Null Handling",
                "description": "Handle null industry sector values with fallbacks"
            },
            {
                "id": "MB_006", 
                "name": "Complex Material Type and Group Logic",
                "segment_id": 577,
                "query": """if MATERIAL TYPE IN ROH, FERT, HALB, then check material group 
if material group in mara is in ('1000', '2000', 'YBMM01') then hard code as 'mat0123'
else check material group in mara_700 and if found in ('L001', 'YBMM01') then hardcode as 'MAT1923'
ELSE DEFAULT TO 'NONE0912'""",
                "expected_type": "Complex Conditional",
                "description": "Complex nested conditional logic for material classification"
            },
            {
                "id": "MB_007",
                "name": "Description Length Calculation",
                "segment_id": 577,
                "query": """For all the Materials of Transformation1, calculate the Length of description of each material and put it in the column "BISMT" and add a new column into target table with name "maktx_temp" if the Length of description is Greater Than 30 hardcode as "G" in the newly added column else hardcode as "L" and finally a response should be returned from LLM as Number of rows(412) affected in the target table""",
                "expected_type": "Complex Calculation",
                "description": "Calculate description length and create conditional columns"
            },
            {
                "id": "MB_008",
                "name": "Delete Column Values",
                "segment_id": 577,
                "query": "Delete the values in the column 'MTART'",
                "expected_type": "Data Deletion",
                "description": "Clear values from specific column"
            },
            {
                "id": "MB_009",
                "name": "Date Field Update",
                "segment_id": 577,
                "query": "For every material in the target table, fetch the LAEDA field from the MARA table (using MATNR as the key), and update the LIQDT field in your target table with this value",
                "expected_type": "Field Update",
                "description": "Update date field from source table"
            },
            {
                "id": "MB_010",
                "name": "Extract Day from Date",
                "segment_id": 577,
                "query": "Add a new column named LIQDT_Date to the target table. For each record, extract only the day part from the LIQDT field (for example, from '20231208' extract '08') and store it in the LIQDT_Date column",
                "expected_type": "Date Extraction",
                "description": "Extract day component from date field"
            },
            {
                "id": "MB_011",
                "name": "Extract Year from Date",
                "segment_id": 577,
                "query": "Add a new column named LIQDT_Year to the target table. For each record, extract only the year part from the LIQDT field (for example, from '20231208' extract '2023') and store it in the LIQDT_Year column",
                "expected_type": "Date Extraction", 
                "description": "Extract year component from date field"
            },
            {
                "id": "MB_012",
                "name": "Extract Quarter from Date",
                "segment_id": 577,
                "query": "Add a new column named LIQDT_Quarter to the target table. For each record, determine the quarter (Q1, Q2, Q3, or Q4) of the date in the LIQDT field and store this value in the LIQDT_Quarter column",
                "expected_type": "Date Calculation",
                "description": "Calculate quarter from date field"
            },
            {
                "id": "MB_013",
                "name": "Calculate Previous Month Last Date",
                "segment_id": 577,
                "query": "Add a new column named LIQDT_LastDate to the target table. For each record, calculate the last date of the previous month based on the value in the LIQDT field, and store this date in the LIQDT_LastDate column",
                "expected_type": "Date Calculation",
                "description": "Calculate last date of previous month"
            },
            {
                "id": "MB_014",
                "name": "Add System Date Column",
                "segment_id": 577,
                "query": "Add a new column to the target table named LIQDT_SystemDate. Populate this column with the system date (i.e., the current date) for all material records in the target table",
                "expected_type": "System Date",
                "description": "Add current system date to all records"
            },
            {
                "id": "MB_015",
                "name": "Extract Day of Week",
                "segment_id": 577,
                "query": "Add a new column into the target table with column name 'LIQDT_Day' and find the Day(Monday, Tuesday, ......) of the date present in the LIQDT field of target table and update in LIQDT_Day field",
                "expected_type": "Date Extraction",
                "description": "Extract day of week from date"
            },
            {
                "id": "MB_016",
                "name": "Delete Column",
                "segment_id": 577,
                "query": "Delete column LIQDT_Day from the target table",
                "expected_type": "Column Deletion",
                "description": "Remove specific column from target table"
            },
            {
                "id": "MB_017",
                "name": "Conditional Value Update",
                "segment_id": 577,
                "query": "If the value of MTART in the target table starts with 'H', then set MTART to 'FERT'",
                "expected_type": "Conditional Update",
                "description": "Update values based on condition"
            },
            
            # Material Plant Segment Tests (592)
            {
                "id": "MP_001",
                "name": "Join Basic with Plant Data",
                "segment_id": 592,
                "query": "Join Material from Basic Segment with Material from MARC segment and Bring Material and Plant field from MARC Table for the plants (1710, 9999)",
                "expected_type": "Join Operation",
                "description": "Join basic material data with plant-specific data"
            },
            
            # Material Description Segment Tests (578)
            {
                "id": "MD_001",
                "name": "Extract Material Descriptions",
                "segment_id": 578,
                "query": "Extract Description from MAKT table for the Materials got from Output of Transformation 1",
                "expected_type": "Cross-Segment",
                "description": "Extract descriptions for materials from previous transformation"
            },
            {
                "id": "MD_002",
                "name": "Remove Special Characters",
                "segment_id": 578,
                "query": "Remove special characters from MAKT Descriptions",
                "expected_type": "Text Processing",
                "description": "Clean special characters from description fields"
            },
            {
                "id": "MD_003",
                "name": "Bring Multiple Descriptions",
                "segment_id": 578,
                "query": "Bring Material description from MAKT table for the materials which you have got in Transformation 1. (ex: if source is having multiple description, you must bring all of them)",
                "expected_type": "Multi-Record Extract",
                "description": "Extract all descriptions for materials including multiple entries"
            },
            
            # Customer Segment Tests
            {
                "id": "CU_001",
                "name": "Customer from Business Partner",
                "segment_id": 578,
                "query": "bring customer FROM but000 for grouping BU_GROUPING = BP03 and check if same number is available from Group (KTOKD = CUST) in KNA1. matching entries should come under customer",
                "expected_type": "Cross-Table Validation",
                "description": "Extract customers with business partner grouping validation"
            },
            {
                "id": "CU_002",
                "name": "Customer Name from Address",
                "segment_id": 463,
                "query": "Bring NAME from ADRC (field NAME1) CONSIDERING ADDRNUMBER FIELD FROM BUT020",
                "expected_type": "Address Lookup",
                "description": "Get customer name from address table"
            },
            {
                "id": "CU_003",
                "name": "Account Group Extraction",
                "segment_id": 463,
                "query": "bring KTOKD from KNA1 Based on Customer Number",
                "expected_type": "Simple Lookup",
                "description": "Extract account group for customers"
            },
            {
                "id": "CU_004",
                "name": "Address Fields Extraction",
                "segment_id": 463,
                "query": "Bring Postal code, City and Country from ADRC (field NAME1) CONSIDERING ADDRNUMBER FIELD FROM BUT020",
                "expected_type": "Multi-Field Address",
                "description": "Extract multiple address components"
            },
            {
                "id": "CU_005",
                "name": "Search Term Logic",
                "segment_id": 463,
                "query": """If NAME1 has only one word place that word as it is in SORTL
else if NAME1 has more than one word skip the first word and start considering from second word and from second word we have to pick only first 20 letters and skip the others finally place it in the SORTL""",
                "expected_type": "Text Processing",
                "description": "Complex text processing logic for search terms"
            },
            {
                "id": "CU_006",
                "name": "Transportation Zone Mapping",
                "segment_id": 463,
                "query": "From Land1_Mapping Table take the transportation_zone value corresponding to the LAND1 value and put into LZONE",
                "expected_type": "Table Mapping",
                "description": "Map transportation zones based on country codes"
            },
            {
                "id": "CU_007",
                "name": "Company Code by Region",
                "segment_id": 463,
                "query": "If LAND1 = European Countries (eg:Germany, Spain....) then put '1310' in BUKRS else if LAND1 = Asian Countries then '1320' else '1330'",
                "expected_type": "Regional Mapping",
                "description": "Set company code based on regional classification"
            },
            {
                "id": "CU_008",
                "name": "Reconciliation Account Logic",
                "segment_id": 463,
                "query": """If country is not DE then put 1100020 in AKONT;
if country = DE and KTOKD != CUST then 1100010;
If Account Group(KTOKD) = CUST and DE put 1100040 in AKONT""",
                "expected_type": "Complex Conditional",
                "description": "Complex logic for reconciliation account assignment"
            },
            {
                "id": "CU_009",
                "name": "Sales Organization by Region",
                "segment_id": 463,
                "query": "If LAND1 = European Countries (eg:Germany, Spain....) then put '1310' in VKORG else if LAND1 = Asian Countries then '1320' else '1330'",
                "expected_type": "Regional Mapping",
                "description": "Set sales organization based on regional classification"
            }
        ]
        
        return test_cases
    
    def run_single_test(self, test_case: Dict[str, Any]) -> Dict[str, Any]:
        """Run a single test case"""
        test_id = test_case["id"]
        test_name = test_case["name"]
        segment_id = test_case["segment_id"]
        query = test_case["query"]
        
        logger.info(f"Running Test {test_id}: {test_name}")
        logger.info(f"Segment ID: {segment_id}")
        logger.info(f"Query: {query[:100]}...")
        
        start_time = time.time()
        result = {
            "test_id": test_id,
            "test_name": test_name,
            "segment_id": segment_id,
            "query": query,
            "expected_type": test_case.get("expected_type", "Unknown"),
            "description": test_case.get("description", ""),
            "start_time": datetime.now().isoformat(),
            "status": "FAILED",
            "sql_query": None,
            "result_type": None,
            "result_summary": None,
            "error_message": None,
            "execution_time": 0,
            "session_id": self.session_id
        }
        
        try:
            # Execute the transformation
            sql_query, query_result, session_id = self.sql_tool.process_sequential_query(
                query=query,
                object_id=self.object_id,
                segment_id=segment_id,
                project_id=self.project_id,
                session_id=self.session_id
            )
            
            # Update session ID if it was created
            if not self.session_id:
                self.session_id = session_id
                logger.info(f"Session created: {self.session_id}")
            
            execution_time = time.time() - start_time
            
            # Process results
            if sql_query is None:
                result["status"] = "FAILED"
                result["error_message"] = str(query_result)
                logger.error(f"Test {test_id} failed: {query_result}")
            else:
                result["status"] = "PASSED"
                result["sql_query"] = sql_query
                result["result_type"] = type(query_result).__name__
                
                # Generate result summary
                if isinstance(query_result, pd.DataFrame):
                    result["result_summary"] = f"DataFrame with {len(query_result)} rows, {len(query_result.columns)} columns"
                    logger.info(f"Test {test_id} passed: {result['result_summary']}")
                elif isinstance(query_result, str):
                    result["result_summary"] = f"String result: {query_result[:100]}..."
                    logger.info(f"Test {test_id} passed: {query_result}")
                else:
                    result["result_summary"] = f"Result type: {type(query_result).__name__}"
                    logger.info(f"Test {test_id} passed: {result['result_summary']}")
                
                self.passed_tests += 1
            
            result["execution_time"] = execution_time
            
        except Exception as e:
            execution_time = time.time() - start_time
            result["status"] = "FAILED" 
            result["error_message"] = f"{type(e).__name__}: {str(e)}"
            result["execution_time"] = execution_time
            result["traceback"] = traceback.format_exc()
            
            logger.error(f"Test {test_id} failed with exception: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            
            self.failed_tests += 1
        
        result["end_time"] = datetime.now().isoformat()
        return result
    
    def run_all_tests(self) -> List[Dict[str, Any]]:
        """Run all defined test cases"""
        logger.info("Starting comprehensive DMTool test suite")
        self.start_time = datetime.now()
        
        test_cases = self.define_test_cases()
        self.total_tests = len(test_cases)
        
        logger.info(f"Total tests to run: {self.total_tests}")
        
        # Group tests by segment for logical execution order
        segments = {}
        for test_case in test_cases:
            segment_id = test_case["segment_id"]
            if segment_id not in segments:
                segments[segment_id] = []
            segments[segment_id].append(test_case)
        
        logger.info(f"Tests grouped by segments: {list(segments.keys())}")
        
        # Run tests in segment order (577, 592, 578, 463)
        segment_order = [577, 592, 578, 463]
        
        for segment_id in segment_order:
            if segment_id in segments:
                logger.info(f"\n{'='*60}")
                logger.info(f"Running tests for Segment {segment_id}")
                logger.info(f"{'='*60}")
                
                for test_case in segments[segment_id]:
                    result = self.run_single_test(test_case)
                    self.test_results.append(result)
                    
                    # Brief pause between tests
                    time.sleep(1)
        
        # Run any remaining tests not in the main segments
        for segment_id, test_cases_list in segments.items():
            if segment_id not in segment_order:
                logger.info(f"\n{'='*60}")
                logger.info(f"Running tests for Segment {segment_id}")
                logger.info(f"{'='*60}")
                
                for test_case in test_cases_list:
                    result = self.run_single_test(test_case)
                    self.test_results.append(result)
                    
                    # Brief pause between tests
                    time.sleep(1)
        
        return self.test_results
    
    def generate_summary_report(self) -> Dict[str, Any]:
        """Generate a comprehensive summary report"""
        end_time = datetime.now()
        total_execution_time = (end_time - self.start_time).total_seconds()
        
        # Calculate statistics
        passed_tests = len([r for r in self.test_results if r["status"] == "PASSED"])
        failed_tests = len([r for r in self.test_results if r["status"] == "FAILED"])
        success_rate = (passed_tests / self.total_tests * 100) if self.total_tests > 0 else 0
        
        # Group results by segment
        segment_stats = {}
        for result in self.test_results:
            segment_id = result["segment_id"]
            if segment_id not in segment_stats:
                segment_stats[segment_id] = {"total": 0, "passed": 0, "failed": 0}
            
            segment_stats[segment_id]["total"] += 1
            if result["status"] == "PASSED":
                segment_stats[segment_id]["passed"] += 1
            else:
                segment_stats[segment_id]["failed"] += 1
        
        # Group results by test type
        type_stats = {}
        for result in self.test_results:
            test_type = result["expected_type"]
            if test_type not in type_stats:
                type_stats[test_type] = {"total": 0, "passed": 0, "failed": 0}
            
            type_stats[test_type]["total"] += 1
            if result["status"] == "PASSED":
                type_stats[test_type]["passed"] += 1
            else:
                type_stats[test_type]["failed"] += 1
        
        # Calculate average execution time
        execution_times = [r["execution_time"] for r in self.test_results if r["execution_time"] > 0]
        avg_execution_time = sum(execution_times) / len(execution_times) if execution_times else 0
        
        summary = {
            "test_run_info": {
                "start_time": self.start_time.isoformat(),
                "end_time": end_time.isoformat(),
                "total_execution_time": total_execution_time,
                "session_id": self.session_id
            },
            "overall_stats": {
                "total_tests": self.total_tests,
                "passed_tests": passed_tests,
                "failed_tests": failed_tests,
                "success_rate": round(success_rate, 2),
                "average_execution_time": round(avg_execution_time, 2)
            },
            "segment_stats": segment_stats,
            "type_stats": type_stats,
            "failed_tests": [
                {
                    "test_id": r["test_id"],
                    "test_name": r["test_name"],
                    "error_message": r["error_message"]
                }
                for r in self.test_results if r["status"] == "FAILED"
            ]
        }
        
        return summary
    
    def save_results(self, filename_prefix: str = "dmtool_test_results"):
        """Save test results to files"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save detailed results as JSON
        detailed_filename = f"{filename_prefix}_detailed_{timestamp}.json"
        with open(detailed_filename, 'w') as f:
            json.dump(self.test_results, f, indent=2, default=str)
        
        # Save summary report
        summary = self.generate_summary_report()
        summary_filename = f"{filename_prefix}_summary_{timestamp}.json"
        with open(summary_filename, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        # Save results as CSV for easy analysis
        csv_filename = f"{filename_prefix}_{timestamp}.csv"
        df = pd.DataFrame(self.test_results)
        df.to_csv(csv_filename, index=False)
        
        logger.info(f"Results saved to:")
        logger.info(f"  - Detailed: {detailed_filename}")
        logger.info(f"  - Summary: {summary_filename}")
        logger.info(f"  - CSV: {csv_filename}")
        
        return detailed_filename, summary_filename, csv_filename
    
    def generate_html_report(self, filename_prefix: str = "dmtool_test_report"):
        """Generate an HTML report for better visualization"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        html_filename = f"{filename_prefix}_{timestamp}.html"
        
        summary = self.generate_summary_report()
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>DMTool Test Results Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
                .summary {{ margin: 20px 0; }}
                .stats {{ display: flex; justify-content: space-around; margin: 20px 0; }}
                .stat-box {{ background-color: #e8f4fd; padding: 15px; border-radius: 5px; text-align: center; }}
                .passed {{ background-color: #d4edda; color: #155724; }}
                .failed {{ background-color: #f8d7da; color: #721c24; }}
                table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                .test-passed {{ background-color: #d4edda; }}
                .test-failed {{ background-color: #f8d7da; }}
                .query-cell {{ max-width: 300px; word-wrap: break-word; }}
                .error-cell {{ max-width: 250px; word-wrap: break-word; color: #721c24; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>DMTool SQL Test Results Report</h1>
                <p><strong>Generated:</strong> {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
                <p><strong>Session ID:</strong> {self.session_id}</p>
                <p><strong>Total Execution Time:</strong> {summary['test_run_info']['total_execution_time']:.2f} seconds</p>
            </div>
            
            <div class="summary">
                <h2>Overall Statistics</h2>
                <div class="stats">
                    <div class="stat-box">
                        <h3>Total Tests</h3>
                        <p>{summary['overall_stats']['total_tests']}</p>
                    </div>
                    <div class="stat-box passed">
                        <h3>Passed</h3>
                        <p>{summary['overall_stats']['passed_tests']}</p>
                    </div>
                    <div class="stat-box failed">
                        <h3>Failed</h3>
                        <p>{summary['overall_stats']['failed_tests']}</p>
                    </div>
                    <div class="stat-box">
                        <h3>Success Rate</h3>
                        <p>{summary['overall_stats']['success_rate']}%</p>
                    </div>
                    <div class="stat-box">
                        <h3>Avg Time</h3>
                        <p>{summary['overall_stats']['average_execution_time']:.2f}s</p>
                    </div>
                </div>
            </div>
            
            <div class="summary">
                <h2>Results by Segment</h2>
                <table>
                    <tr>
                        <th>Segment ID</th>
                        <th>Total Tests</th>
                        <th>Passed</th>
                        <th>Failed</th>
                        <th>Success Rate</th>
                    </tr>"""
        
        for segment_id, stats in summary['segment_stats'].items():
            success_rate = (stats['passed'] / stats['total'] * 100) if stats['total'] > 0 else 0
            html_content += f"""
                    <tr>
                        <td>{segment_id}</td>
                        <td>{stats['total']}</td>
                        <td class="passed">{stats['passed']}</td>
                        <td class="failed">{stats['failed']}</td>
                        <td>{success_rate:.1f}%</td>
                    </tr>"""
        
        html_content += """
                </table>
            </div>
            
            <div class="summary">
                <h2>Results by Test Type</h2>
                <table>
                    <tr>
                        <th>Test Type</th>
                        <th>Total Tests</th>
                        <th>Passed</th>
                        <th>Failed</th>
                        <th>Success Rate</th>
                    </tr>"""
        
        for test_type, stats in summary['type_stats'].items():
            success_rate = (stats['passed'] / stats['total'] * 100) if stats['total'] > 0 else 0
            html_content += f"""
                    <tr>
                        <td>{test_type}</td>
                        <td>{stats['total']}</td>
                        <td class="passed">{stats['passed']}</td>
                        <td class="failed">{stats['failed']}</td>
                        <td>{success_rate:.1f}%</td>
                    </tr>"""
        
        html_content += """
                </table>
            </div>
            
            <div class="summary">
                <h2>Detailed Test Results</h2>
                <table>
                    <tr>
                        <th>Test ID</th>
                        <th>Test Name</th>
                        <th>Segment</th>
                        <th>Status</th>
                        <th>Query</th>
                        <th>Result Summary</th>
                        <th>Execution Time</th>
                        <th>Error Message</th>
                    </tr>"""
        
        for result in self.test_results:
            status_class = "test-passed" if result['status'] == 'PASSED' else "test-failed"
            query_preview = result['query'][:100] + "..." if len(result['query']) > 100 else result['query']
            error_msg = result.get('error_message', '')[:150] + "..." if result.get('error_message', '') and len(result.get('error_message', '')) > 150 else result.get('error_message', '')
            
            html_content += f"""
                    <tr class="{status_class}">
                        <td>{result['test_id']}</td>
                        <td>{result['test_name']}</td>
                        <td>{result['segment_id']}</td>
                        <td>{result['status']}</td>
                        <td class="query-cell">{query_preview}</td>
                        <td>{result.get('result_summary', 'N/A')}</td>
                        <td>{result['execution_time']:.2f}s</td>
                        <td class="error-cell">{error_msg}</td>
                    </tr>"""
        
        html_content += """
                </table>
            </div>
        </body>
        </html>"""
        
        with open(html_filename, 'w') as f:
            f.write(html_content)
        
        logger.info(f"HTML report saved to: {html_filename}")
        return html_filename
    
    def print_summary(self):
        """Print a summary of test results to console"""
        summary = self.generate_summary_report()
        
        print("\n" + "="*80)
        print("DMTool SQL Test Suite - SUMMARY REPORT")
        print("="*80)
        
        print(f"\nTest Run Information:")
        print(f"  Start Time: {summary['test_run_info']['start_time']}")
        print(f"  End Time: {summary['test_run_info']['end_time']}")
        print(f"  Total Execution Time: {summary['test_run_info']['total_execution_time']:.2f} seconds")
        print(f"  Session ID: {summary['test_run_info']['session_id']}")
        
        print(f"\nOverall Statistics:")
        print(f"  Total Tests: {summary['overall_stats']['total_tests']}")
        print(f"  Passed Tests: {summary['overall_stats']['passed_tests']}")
        print(f"  Failed Tests: {summary['overall_stats']['failed_tests']}")
        print(f"  Success Rate: {summary['overall_stats']['success_rate']}%")
        print(f"  Average Execution Time: {summary['overall_stats']['average_execution_time']:.2f} seconds")
        
        print(f"\nResults by Segment:")
        for segment_id, stats in summary['segment_stats'].items():
            success_rate = (stats['passed'] / stats['total'] * 100) if stats['total'] > 0 else 0
            print(f"  Segment {segment_id}: {stats['passed']}/{stats['total']} passed ({success_rate:.1f}%)")
        
        print(f"\nResults by Test Type:")
        for test_type, stats in summary['type_stats'].items():
            success_rate = (stats['passed'] / stats['total'] * 100) if stats['total'] > 0 else 0
            print(f"  {test_type}: {stats['passed']}/{stats['total']} passed ({success_rate:.1f}%)")
        
        if summary['failed_tests']:
            print(f"\nFailed Tests:")
            for failed_test in summary['failed_tests']:
                print(f"  {failed_test['test_id']}: {failed_test['test_name']}")
                print(f"    Error: {failed_test['error_message'][:100]}...")
        
        print("\n" + "="*80)


def main():
    """Main function to run the test suite"""
    print("DMTool SQL Comprehensive Test Suite")
    print("="*50)
    
    # Initialize tester
    tester = DMToolTester()
    
    try:
        # Run all tests
        print("Starting test execution...")
        results = tester.run_all_tests()
        
        # Print summary to console
        tester.print_summary()
        
        # Save results to files
        print("\nSaving results...")
        detailed_file, summary_file, csv_file = tester.save_results()
        
        # Generate HTML report
        print("Generating HTML report...")
        html_file = tester.generate_html_report()
        
        print(f"\nTest suite completed!")
        print(f"Files generated:")
        print(f"  - Detailed JSON: {detailed_file}")
        print(f"  - Summary JSON: {summary_file}")
        print(f"  - CSV Report: {csv_file}")
        print(f"  - HTML Report: {html_file}")
        
        # Return summary for programmatic use
        return tester.generate_summary_report()
        
    except Exception as e:
        logger.error(f"Test suite failed with error: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise
    
    finally:
        # Cleanup if needed
        print("\nTest suite execution completed.")


if __name__ == "__main__":
    # Run the test suite
    try:
        summary = main()
        
        # Exit with appropriate code
        if summary['overall_stats']['failed_tests'] > 0:
            print(f"\nWarning: {summary['overall_stats']['failed_tests']} tests failed.")
            sys.exit(1)
        else:
            print(f"\nAll {summary['overall_stats']['total_tests']} tests passed successfully!")
            sys.exit(0)
            
    except KeyboardInterrupt:
        print("\nTest suite interrupted by user.")
        sys.exit(2)
    except Exception as e:
        print(f"\nTest suite failed with error: {e}")
        sys.exit(3)