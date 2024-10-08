

class CypherQuery:
    def __init__(self,graph):
        self.graph = graph

    def set_entity(self,uuid=None):
        self.graph.query(
            f"""
                MATCH (n:`__Entity__`)
                REMOVE n:`__Entity__`
                SET n:`__Entity__{uuid}`
            """
        )
        
    def set_document(self,uuid=None):
        self.graph.query(
            f"""
                MATCH (n:`Document`)
                REMOVE n:`Document`
                SET n:`Document{uuid}`
            """
        )
    
    # FIXME: 注册gds.graph时也要加上uuid,不然可能导致多进程误删除
    def drop_entites(self):
        # 删除名字为entities的图
        try:
            self.graph.query(
                """
                CALL gds.graph.drop('entities')
                """
            )
        except:
            print("`entities` does not exist")
    
    # FIXME: 注册gds.graph时也要加上uuid,不然可能导致多进程误删除       
    def drop_communities(self):        
        try:
            self.graph.query(
                f"""
                CALL gds.graph.drop('communities')
                """
            )
        except:
            print("`communities` does not exist")
            
    # 社区检测与聚类分析
    def detect(self,uuid,word_edit_distance):
        return self.graph.query(
            f"""MATCH (e:`__Entity__{uuid}`)
            WHERE size(e.id) > 4 // longer than 4 characters
            WITH e.wcc AS community, collect(e) AS nodes, count(*) AS count
            WHERE count > 1
            UNWIND nodes AS node
            // Add text distance
            WITH distinct
            [n IN nodes WHERE apoc.text.distance(toLower(node.id), toLower(n.id)) < $distance | n.id] AS intermediate_results
            WHERE size(intermediate_results) > 1
            WITH collect(intermediate_results) AS results
            // combine groups together if they share elements
            UNWIND range(0, size(results)-1, 1) as index
            WITH results, index, results[index] as result
            WITH apoc.coll.sort(reduce(acc = result, index2 IN range(0, size(results)-1, 1) |
                    CASE WHEN index <> index2 AND
                        size(apoc.coll.intersection(acc, results[index2])) > 0
                        THEN apoc.coll.union(acc, results[index2])
                        ELSE acc
                    END
            )) as combinedResult
            WITH distinct(combinedResult) as combinedResult
            // extra filtering
            WITH collect(combinedResult) as allCombinedResults
            UNWIND range(0, size(allCombinedResults)-1, 1) as combinedResultIndex
            WITH allCombinedResults[combinedResultIndex] as combinedResult, combinedResultIndex, allCombinedResults
            WHERE NOT any(x IN range(0,size(allCombinedResults)-1,1)
                WHERE x <> combinedResultIndex
                AND apoc.coll.containsAll(allCombinedResults[x], combinedResult)
            )
            RETURN combinedResult
            """, params={'distance': word_edit_distance}
        )
        
    def add_constraints_for_community(self,uuid=None):
        self.graph.query(f"CREATE CONSTRAINT IF NOT EXISTS FOR (c:`__Community__{uuid}`) REQUIRE c.id IS UNIQUE;")
    
    # 构造层次聚类
    def constructing_hierarchical_clustering(self,uuid=None):
        return self.graph.query("""
                MATCH (e:`{entity_label}`)
                UNWIND range(0, size(e.communities) - 1 , 1) AS index
                CALL {{
                WITH e, index
                WITH e, index
                WHERE index = 0
                MERGE (c:`{community_label}` {{id: toString(index) + '-' + toString(e.communities[index])}})
                ON CREATE SET c.level = index
                MERGE (e)-[:IN_COMMUNITY]->(c)
                RETURN count(*) AS count_0
                }}
                CALL {{
                WITH e, index
                WITH e, index
                WHERE index > 0
                MERGE (current:`{community_label}` {{id: toString(index) + '-' + toString(e.communities[index])}})
                ON CREATE SET current.level = index
                MERGE (previous:`{community_label}` {{id: toString(index - 1) + '-' + toString(e.communities[index - 1])}})
                ON CREATE SET previous.level = index - 1
                MERGE (previous)-[:IN_COMMUNITY]->(current)
                RETURN count(*) AS count_1
                }}
                RETURN count(*)
            """.format(entity_label=f"__Entity__{uuid}",community_label=f"__Community__{uuid}")
        )
    
    def set_community_rank(self,uuid=None):
        self.graph.query(f"""
            MATCH (c:`__Community__{uuid}`)<-[:IN_COMMUNITY*]-(:`__Entity__{uuid}`)<-[:MENTIONS]-(d:`Document{uuid}`) // 匹配社区文档
            WITH c, count(distinct d) AS rank   //  计算每个社区包含的不同的文档数量作为社区的排名
            SET c.community_rank = rank;    // 设置社区排名
            """
        )
        
    def set_node_degree(self,uuid=None):
        node_degree_query = f"""
            MATCH (n)
            WHERE ANY(label IN labels(n) WHERE label ENDS WITH '{uuid}')
            SET n.degree = apoc.node.degree(n)
            RETURN count(n) AS modified_nodes;
        """
        self.graph.query(node_degree_query)
    
    def set_relationship_degree(self,uuid=None):
        relationship_degree_query = f"""
            MATCH (n) 
            WHERE n.degree is not NULL and ANY(label IN labels(n) WHERE label ENDS WITH '{uuid}')
            WITH n as source
            MATCH (source)-[r]->(target)
            WHERE target.degree is not null and ANY(label IN labels(target) WHERE label ENDS WITH '{uuid}')
            SET r.source_degree=source.degree,r.target_degree=target.degree,r.rank=source.degree+target.degree
            RETURN COUNT(r) AS modified_relationships; // 返回被修改的边的数量
        """
        self.graph.query(relationship_degree_query)
    
    def set_text_unit_ids(self,uuid=None):
        text_unit_ids_query = f"""
            MATCH (n:`__Entity__{uuid}`)
            MATCH (p:`Document{uuid}`)
            WHERE p.text IS NOT NULL
            WITH n, collect(p) AS text_units
            UNWIND text_units AS text_unit
            WITH n,text_unit
            WHERE text_unit.text CONTAINS n.id  // 使用 CONTAINS 检查
            WITH n, collect(text_unit.id) AS text_unit_ids
            SET n.text_unit_ids = text_unit_ids
            RETURN count(DISTINCT n) AS modified_nodes;
        """
        self.graph.query(text_unit_ids_query)
    
    def set_relationship_source_and_target(self,uuid=None):
        self.graph.query(
            f"""
            MATCH (n:`__Entity__{uuid}`)-[r]->(m:`__Entity__{uuid}`)
            WITH n,r,m
            SET r.source = n.id, r.target = m.id
            RETURN count(r) AS modified_relationships
            """
        )
    
    def set_communities(self,uuid=None):
        self.graph.query(
            f"""
            MATCH (n:`__Entity__{uuid}`)-[:IN_COMMUNITY*]->(c:`__Community__{uuid}`)
            WITH n, collect(c.id) AS community_ids
            SET n.communities = community_ids
            RETURN count(n) AS modified_nodes;
            """
        )
    
    def get_community_info(self,user_id=None):
        return self.graph.query("""
            MATCH (c:`{community_label}`)<-[:IN_COMMUNITY*]-(e:`{entity_label}`) // 匹配社区实体
            // WHERE c.level in [1]
            WITH c, collect(e) AS nodes
            WHERE size(nodes) > 1
            CALL apoc.path.subgraphAll(nodes[0], {{
                whitelistNodes:nodes
            }})
            YIELD relationships
            RETURN c.id AS communityId,
                [n in nodes | {{id: n.id, description: n.description, type: [el in labels(n) WHERE el <> '{entity_label}'][0]}}] AS nodes,
                [r in relationships | {{start: startNode(r).id, type: type(r), end: endNode(r).id, description: r.description}}] AS rels
            """.format(entity_label=f"__Entity__{user_id}", community_label=f"__Community__{user_id}")
        )
    
    def store_info(self,info,uuid=None):
        self.graph.query(
            f"""
                UNWIND $info AS info
                MATCH (c:`__Community__{uuid}` {{id: info.community}})
                SET c.summary = info.summary,c.title = info.title
            """, params={"info": info}
        )