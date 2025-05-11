"""
Example usage of the TableLLM Relational Model.

This script demonstrates how to use the relational model for multi-segment processing.
"""

import pandas as pd
from relational_integration import TableLLMRelational

def main():
    print("TableLLM Relational Model Example")
    print("==================================")
    
    # Initialize the TableLLM relational interface
    table_llm = TableLLMRelational()
    
    # Set up parameters
    object_id = 41
    project_id = 24
    session_id = None  # Will be created in first query
    
    print("\n1. First segment (336) - Initial query")
    query1 = "Get material type from MARA and map it to MTART in the target table"
    target_field1 = "MTART"
    segment_id1 = 577
    
    code1, result1, session_id = table_llm.process_query(
        query1, object_id, segment_id1, project_id, 
        session_id=session_id,
        target_sap_fields=target_field1
    )
    
    print(f"Session ID: {session_id}")
    print(f"Result shape: {result1.shape if isinstance(result1, pd.DataFrame) else 'Not a DataFrame'}")
    print(f"Result sample:\n{result1.head(3) if isinstance(result1, pd.DataFrame) else result1}")
    
    print("\n2. Same segment (336) - Follow-up query")
    query2 = "Now add the material description from MAKT where the language is EN"
    target_field2 = "MAKTX"
    
    code2, result2, session_id = table_llm.process_query(
        query2, object_id, segment_id1, project_id, 
        session_id=session_id,
        target_sap_fields=target_field2
    )
    
    print(f"Result shape: {result2.shape if isinstance(result2, pd.DataFrame) else 'Not a DataFrame'}")
    print(f"Result sample:\n{result2.head(3) if isinstance(result2, pd.DataFrame) else result2}")
    
    print("\n3. New segment (337) - New table query")
    query3 = "Create a new table with vendor information from LFA1 and get their names"
    target_field3 = "NAME1"
    segment_id2 = 578
    
    code3, result3, session_id = table_llm.process_query(
        query3, object_id, segment_id2, project_id, 
        session_id=session_id,
        target_sap_fields=target_field3
    )
    
    print(f"Result shape: {result3.shape if isinstance(result3, pd.DataFrame) else 'Not a DataFrame'}")
    print(f"Result sample:\n{result3.head(3) if isinstance(result3, pd.DataFrame) else result3}")
    
    print("\n4. Back to first segment (336) - Switch back")
    query4 = "Add the base unit of measure from MARA as MEINS"
    target_field4 = "MEINS"
    
    code4, result4, session_id = table_llm.process_query(
        query4, object_id, segment_id1, project_id, 
        session_id=session_id,
        target_sap_fields=target_field4
    )
    
    print(f"Result shape: {result4.shape if isinstance(result4, pd.DataFrame) else 'Not a DataFrame'}")
    print(f"Result sample:\n{result4.head(3) if isinstance(result4, pd.DataFrame) else result4}")
    
    # Session information and visualization
    print("\n5. Session Information")
    session_info = table_llm.get_session_info(session_id)
    print(f"Session info: {session_info}")
    
    # Generate ER diagram
    print("\n6. Session Diagram")
    diagram = table_llm.get_session_diagram(session_id)
    print(diagram)
    
    # Get relationships
    print("\n7. Session Relationships")
    relationships = table_llm.get_session_relationships(session_id)
    for rel in relationships:
        print(f"Relationship: {rel['from_table']} -> {rel['to_table']} ({rel['type']})")
        print(f"Mapping: {rel['mapping']}")
        print("-" * 40)
    
    # Validate session integrity
    print("\n8. Session Integrity Validation")
    validation = table_llm.validate_session_integrity(session_id)
    print(f"Valid: {validation.get('valid', False)}")
    if not validation.get('valid', False):
        if 'tables' in validation:
            for table_name, result in validation['tables'].items():
                if not result.get('valid', True):
                    print(f"Table {table_name} has issues:")
                    for issue in result.get('issues', []):
                        print(f"  - {issue}")

if __name__ == "__main__":
    main()
