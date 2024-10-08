
import argparse
from langchain_community.graphs import Neo4jGraph

def parse_args():
    arg_parser = argparse.ArgumentParser(description="Drop Something")
    
    arg_parser.add_argument("--neo4j_uri", type=str,default=None, help="Neo4j URI")
    arg_parser.add_argument("--neo4j_username", type=str,default=None, help="Neo4j user")
    arg_parser.add_argument("--neo4j_password", type=str,default=None, help="Neo4j password")
    arg_parser.add_argument("--uuid", type=str, default="", help="UUID for the index")
    
    return arg_parser.parse_args()

def drop():
    args = parse_args()
    graph = Neo4jGraph(uri=args.neo4j_uri, user=args.neo4j_username, password=args.neo4j_password)

    graph.query(f"DROP INDEX `{args.uuid}` IF EXISTS")
    graph.query(f"DROP CONSTRAINT ON (n:`__Entity__{args.uuid}`) ASSERT n.id IS UNIQUE")    
    graph.query(f"MATCH (n:`__Entity__{args.uuid}`) DETACH DELETE n")