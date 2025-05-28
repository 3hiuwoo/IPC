import json
import os
import threading

from tqdm import tqdm

from graph import graph
from llm import llm, embeddings


class MedicalKnowledgeGraphBuilder:
    """
    A class to build a medical knowledge graph from JSON data
    and store it in Neo4j database using langchain_neo4j.
    """
    def __init__(self):
        """Initialize the knowledge graph builder with empty entity and relation lists."""
        self.graph = graph

        # Entity nodes (8 types)
        self.drugs = []          # Drugs
        self.recipes = []        # Recipes
        self.foods = []          # Foods
        self.checks = []         # Medical checks
        self.departments = []    # Medical departments
        self.producers = []      # Drug manufacturers
        self.diseases = []       # Diseases
        self.symptoms = []       # Symptoms

        self.disease_infos = []  # Disease information
        self.disease_properties = ['desc', 'prevent', 'cause', 'get_prob', 'easy_get', 'cure_way', 'cure_lasttime', 'cured_prob']  # Disease properties
        # Relationship edges
        self.rels_department = []       # Department-Department relations
        self.rels_not_eat = []          # Disease-Forbidden food relations
        self.rels_do_eat = []           # Disease-Recommended food relations
        self.rels_recommend_eat = []    # Disease-Recommended recipe relations
        self.rels_common_drug = []      # Disease-Common drug relations
        self.rels_recommend_drug = []   # Disease-Recommended drug relations
        self.rels_check = []            # Disease-Check relations
        self.rels_drug_producer = []    # Manufacturer-Drug relations
        self.rels_symptom = []          # Disease-Symptom relations
        self.rels_accompany = []        # Disease-Accompanying disease relations
        self.rels_category = []         # Disease-Department relations
        
        
    def extract_triples(self, path):
        """
        Extract entity and relationship triples from JSON file.
        
        Args:
            data_path: Path to the JSON data file
        """
        
        with open(path, 'r', encoding='utf8') as f:
            for line in tqdm(f.readlines(), desc="Extracting triples from JSON"):
                data_json = json.loads(line)
                disease_dict = {}
                disease = data_json['name']
                disease_dict['name'] = disease
                self.diseases.append(disease)
                
                # Initialize disease attributes
                disease_dict.update({k: '' for k in self.disease_properties})

                # Process symptoms
                if 'symptom' in data_json:
                    self.symptoms.extend(data_json['symptom'])
                    for symptom in data_json['symptom']:
                        self.rels_symptom.append([disease, 'has_symptom', symptom])

                # Process accompanying diseases
                if 'acompany' in data_json:
                    for accompany in data_json['acompany']:
                        self.rels_accompany.append([disease, 'accompany_with', accompany])
                        self.diseases.append(accompany)

                # Process disease descriptions and attributes
                for key in self.disease_properties:
                    if key in data_json:
                        disease_dict[key] = data_json[key]

                # Process cure departments
                if 'cure_department' in data_json:
                    cure_department = data_json['cure_department']
                    if len(cure_department) == 1:
                        self.rels_category.append([disease, 'cure_department', cure_department[0]])
                    if len(cure_department) == 2:
                        parent = cure_department[0]
                        child = cure_department[1]
                        self.rels_department.append([child, 'belongs_to', parent])
                        self.rels_category.append([disease, 'cure_department', child])

                    disease_dict['cure_department'] = cure_department
                    self.departments.extend(cure_department)

                # Process drugs
                if 'common_drug' in data_json:
                    common_drug = data_json['common_drug']
                    for drug in common_drug:
                        self.rels_common_drug.append([disease, 'has_common_drug', drug])
                    self.drugs.extend(common_drug)

                if 'recommand_drug' in data_json:
                    recommend_drug = data_json['recommand_drug']
                    self.drugs.extend(recommend_drug)
                    for drug in recommend_drug:
                        self.rels_recommend_drug.append([disease, 'recommend_drug', drug])

                # Process diet information
                if 'not_eat' in data_json:
                    not_eat = data_json['not_eat']
                    for food in not_eat:
                        self.rels_not_eat.append([disease, 'not_eat', food])
                    self.foods.extend(not_eat)
                    
                if 'do_eat' in data_json:
                    do_eat = data_json['do_eat']
                    for food in do_eat:
                        self.rels_do_eat.append([disease, 'do_eat', food])
                    self.foods.extend(do_eat)

                if 'recommand_eat' in data_json:
                    recommend_eat = data_json['recommand_eat']
                    for recipe in recommend_eat:
                        self.rels_recommend_eat.append([disease, 'recommend_recipes', recipe])
                    self.recipes.extend(recommend_eat)

                # Process medical checks
                if 'check' in data_json:
                    checks = data_json['check']
                    for check in checks:
                        self.rels_check.append([disease, 'need_check', check])
                    self.checks.extend(checks)

                # Process drug details
                if 'drug_detail' in data_json:
                    for detail in data_json['drug_detail']:
                        parts = detail.split('(')
                        if len(parts) == 2:
                            producer, drug = parts
                            drug = drug.rstrip(')')
                            if producer.find(drug) > 0:
                                producer = producer.rstrip(drug)
                            self.producers.append(producer)
                            self.drugs.append(drug)
                            self.rels_drug_producer.append([producer, 'production', drug])
                        else:
                            drug = parts[0]
                            self.drugs.append(drug)

                self.disease_infos.append(disease_dict)


    def create_nodes(self, entities, entity_type):
        """
        Create nodes in Neo4j for the given entities.
        
        Args:
            entities: List of entity names
            entity_type: Type label for the entities
        """
        
        for node in tqdm(set(entities), desc=f"Creating {entity_type} nodes"):
            # Use MERGE to avoid duplicates
            cypher = f"""
            MERGE (n:{entity_type} {{name: $name}})
            """
            try:
                # In langchain_neo4j, we use query instead of run, and parameters instead of string formatting
                self.graph.query(cypher, {"name": node.replace("'", "")})
            except Exception as e:
                print(f"Error creating node: {e}")
                print(f"Failed query: {cypher} with name={node}")
        
        
    def create_relationships(self, triples, source_type, target_type):
        """
        Create relationships in Neo4j for the given triples.
        
        Args:
            triples: List of [source, relation, target] triples
            source_type: Entity type for the source node
            target_type: Entity type for the target node
        """
        
        if not triples:
            return
            
        relation_type = triples[0][1]
        
        for source, relation, target in tqdm(triples, desc=f"Creating {relation_type} relationships"):
            cypher = f"""
            MATCH (s:{source_type}), (t:{target_type})
            WHERE s.name = $source AND t.name = $target
            MERGE (s)-[r:{relation}]->(t)
            """
            try:
                self.graph.query(cypher, {
                    "source": source.replace("'", ""), 
                    "target": target.replace("'", "")
                })
            except Exception as e:
                print(f"Error creating relationship: {e}")
                print(f"Failed query: {cypher}")


    def set_node_properties(self, entity_infos, entity_type):
        """
        Set properties for nodes in Neo4j.
        
        Args:
            entity_infos: List of dictionaries containing entity information
            entity_type: Type of entity to update
        """
        
        for entity_dict in tqdm(entity_infos, desc=f"Setting {entity_type} properties"):
            name = entity_dict['name']
            properties = {k: v for k, v in entity_dict.items() if k != 'name'}
            
            # Convert list properties to string to avoid type issues
            for k, v in properties.items():
                # if isinstance(v, list):
                #     properties[k] = v
                if isinstance(v, str):
                    properties[k] = v.replace("'", "").replace("\n", " ")
            
            # Set all properties in one query
            if properties:
                set_clauses = ", ".join([f"n.{k} = ${k}" for k in properties.keys()])
                cypher = f"""
                MATCH (n:{entity_type})
                WHERE n.name = $name
                SET {set_clauses}
                """
                try:
                    params = {"name": name.replace("'", "")}
                    params.update(properties)
                    self.graph.query(cypher, params)
                except Exception as e:
                    print(f"Error setting attributes: {e}")
                    print(f"Failed query: {cypher}")


    def build_nodes(self):
        """Create all entity nodes in the knowledge graph."""
        
        entity_types = [
            (self.drugs, "Drug"),
            (self.recipes, "Recipe"),
            (self.foods, "Food"),
            (self.checks, "Check"),
            (self.departments, "Department"),
            (self.producers, "Producer"),
            (self.diseases, "Disease"),
            (self.symptoms, "Symptom")
        ]
        
        for entities, entity_type in entity_types:
            self.create_nodes(entities, entity_type)


    def build_relationships(self):
        """Create all relationships in the knowledge graph."""
        
        relation_types = [
            (self.rels_department, "Department", "Department"),
            (self.rels_not_eat, "Disease", "Food"),
            (self.rels_do_eat, "Disease", "Food"),
            (self.rels_recommend_eat, "Disease", "Recipe"),
            (self.rels_common_drug, "Disease", "Drug"),
            (self.rels_recommend_drug, "Disease", "Drug"),
            (self.rels_check, "Disease", "Check"),
            (self.rels_drug_producer, "Producer", "Drug"),
            (self.rels_symptom, "Disease", "Symptom"),
            (self.rels_accompany, "Disease", "Disease"),
            (self.rels_category, "Disease", "Department")
        ]
        
        for triples, source_type, target_type in relation_types:
            self.create_relationships(triples, source_type, target_type)


    def set_disease_properties(self):
        """Set properties for disease nodes using threading to avoid timeout."""
        
        thread = threading.Thread(
            target=self.set_node_properties,
            args=(self.disease_infos, "Disease")
        )
        thread.daemon = False
        thread.start()

    
    def export_json(self, data, path):
        """
        Export data as JSON to a specified file path.
        
        Args:
            data: List of items to export
            path: File path to save the JSON data
        """
        
        print(f"Exporting data to {path}")
        if isinstance(data[0], str):
            data = sorted([d.strip("...") for d in set(data)])
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)


    def export(self, data_path):
        """
        Export all entities and relationships to JSON files.
        Creates separate files for each entity type and relationship type.
        """
        
        # Create base directory
        os.makedirs(data_path, exist_ok=True)
        print("Exporting entities and relations to JSON files")
        
        # Export entities
        entity_exports = [
            (self.drugs, os.path.join(data_path, 'drugs.json')),
            (self.recipes, os.path.join(data_path, 'recipes.json')),
            (self.foods, os.path.join(data_path, 'foods.json')),
            (self.checks, os.path.join(data_path, 'checks.json')),
            (self.departments, os.path.join(data_path, 'departments.json')),
            (self.producers, os.path.join(data_path, 'producers.json')),
            (self.diseases, os.path.join(data_path, 'diseases.json')),
            (self.symptoms, os.path.join(data_path, 'symptoms.json'))
        ]
        
        for data, path in entity_exports:
            self.export_json(data, path)
        
        # Export relations
        relation_exports = [
            (self.rels_department, os.path.join(data_path, 'rels_department.json')),
            (self.rels_not_eat, os.path.join(data_path, 'rels_not_eat.json')),
            (self.rels_do_eat, os.path.join(data_path, 'rels_do_eat.json')),
            (self.rels_recommend_eat, os.path.join(data_path, 'rels_recommend_eat.json')),
            (self.rels_common_drug, os.path.join(data_path, 'rels_common_drug.json')),
            (self.rels_recommend_drug, os.path.join(data_path, 'rels_recommend_drug.json')),
            (self.rels_check, os.path.join(data_path, 'rels_check.json')),
            (self.rels_drug_producer, os.path.join(data_path, 'rels_drug_producer.json')),
            (self.rels_symptom, os.path.join(data_path, 'rels_symptom.json')),
            (self.rels_accompany, os.path.join(data_path, 'rels_accompany.json')),
            (self.rels_category, os.path.join(data_path, 'rels_category.json'))
        ]
        
        for data, path in relation_exports:
            self.export_json(data, path)
        
        print("Export completed successfully!")
    
        
    def build(self, path):
        """
        Build the entire knowledge graph by extracting triples,
        creating nodes, relationships, and setting attributes.
        """
        
        print("Building medical knowledge graph")
        
        # Extract triples from JSON data
        self.extract_triples(path)
        self.build_nodes()
        self.set_disease_properties()
        self.build_relationships()
        
        print("Knowledge graph built successfully!")
        

if __name__ == '__main__':
    data_path = "./data/medical.json"
    kg_builder = MedicalKnowledgeGraphBuilder()
    kg_builder.build(data_path)
    kg_builder.export("./data/summary")