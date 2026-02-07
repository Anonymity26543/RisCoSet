from config.typing import *


def SolutionPrompt(task: Problem) -> str:
    return f"""
    {task["description"]}
    """


def TestcasePrompt(testcase: TestCase) -> str:
    pass


def InputgenPrompt(task: Problem) -> str:
    return f"""
    # Please complete the function *input_generator* based on the following description:
    \"\"\"
     Generate multiple valid, diverse, and varied-complexity inputs for {task["entry_point"]} according to the following requirements:
        1. Carefully analyze to understand the functionality and purpose of {task["entry_point"]} the infer the types of inputs the code would reasonably expect based on its description and behavior.
        2. Create inputs that cover a wide range of possibilities the code might encounter during execution.
        3. All generated inputs must be valid in the context of the target code’s functionality.
        4. Vary difficulty in the size, structure, values, and complexity of the inputs.
        5. The input_generator function should return a list of inputs, where each input is represented appropriately (e.g., integers, strings, lists, etc.).
    \"\"\"
    \n\n\n
    
    # Below is a description of the function {task["entry_point"]}.\n
    \"\"\"
    {task["description"]}
    \"\"\" \n
    
    # Please implement the function *input_generator* based on the signature format: 
    # def input_generator() -> List[Any]:
    # Your code here\n
        
    """
