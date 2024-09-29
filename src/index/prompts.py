from dataclasses import field, dataclass

@dataclass
class SystemPrompts:
    """Prompts for the graphrag algorithm"""
    GRAPHSYSTEMPROMPT:str = field(default="# Knowledge Graph Instructions for {model_name}\n"
                                "## 1. Overview\n"
                                "You are a top-tier algorithm designed for extracting information in structured "
                                "formats to build a knowledge graph.\n"
                                "Try to capture as much information from the text as possible without "
                                "sacrificing accuracy. Do not add any information that is not explicitly "
                                "mentioned in the text.\n"
                                "- **Nodes** represent entities and concepts.\n"
                                "- The aim is to achieve simplicity and clarity in the knowledge graph, making it\n"
                                "accessible for a vast audience.\n"
                                "## 2. Labeling Nodes\n"
                                "- **Consistency**: Ensure you use available types for node labels.\n"
                                "Ensure you use basic or elementary types for node labels.\n"
                                "- For example, when you identify an entity representing a person, "
                                "always label it as **'person'**. Avoid using more specific terms "
                                "like 'mathematician' or 'scientist'."
                                "- **Node IDs**: Never utilize integers as node IDs. Node IDs should be "
                                "names or human-readable identifiers found in the text.\n"
                                "- **Relationships** represent connections between entities or concepts.\n"
                                "Ensure consistency and generality in relationship types when constructing "
                                "knowledge graphs. Instead of using specific and momentary types "
                                "such as 'BECAME_PROFESSOR', use more general and timeless relationship types "
                                "like 'PROFESSOR'. Make sure to use general and timeless relationship types!\n"
                                "## 3. Coreference Resolution\n"
                                "- **Maintain Entity Consistency**: When extracting entities, it's vital to "
                                "ensure consistency.\n"
                                'If an entity, such as "John Doe", is mentioned multiple times in the text '
                                'but is referred to by different names or pronouns (e.g., "Joe", "he"),'
                                "always use the most complete identifier for that entity throughout the "
                                'knowledge graph. In this example, use "John Doe" as the entity ID.\n'
                                "Remember, the knowledge graph should be coherent and easily understandable, "
                                "so maintaining consistency in entity references is crucial.\n"
                                "## 4. Strict Compliance\n"
                                "Adhere to the rules strictly. Non-compliance will result in termination."
                            )
    IDENTIFY_SYSTEM_PROMPT:str = field(default="""You are a data processing assistant. Your task is to identify duplicate entities in a list and decide which of them should be merged.
                                The entities might be slightly different in format or content, but essentially refer to the same thing. Use your analytical skills to determine duplicates.

                                Here are the rules for identifying duplicates:
                                1. Entities with minor typographical differences should be considered duplicates.
                                2. Entities with different formats but the same content should be considered duplicates.
                                3. Entities that refer to the same real-world object or concept, even if described differently, should be considered duplicates.
                                4. If it refers to different numbers, dates, or products, do not merge results
                                """
                                )

@dataclass
class UserPrompts:
    GRAPH_USER_PROMPT:str = field(default=(
        "Tip: Make sure to answer in the correct format and do "
        "not include any explanations. "
        "Use the given format to extract information from the "
        "following input: {input}"
    ))

    IDENTIFY_USER_PROMPT:str = field(default="""
                                Here is the list of entities to process:
                                {entities}

                                Please identify duplicates, merge them, and provide the merged list.
                                """
                                )
