""" model_write_class

        The model_write_class provides methods for writing custom ollama modelfile's through automation.
    This process allows for on the spot model creation with defined parameters for any given model or agent
    build/workflow. 

  File "D:\CodingGit_StorageHDD\Ollama_Custom_Mods\ollama_agent_roll_cage\ollama_mod_cage\wizard_spell_book\Public_Chatbot_Base_Wand\write_modelfile\model_write_class.py", line 22, in __init__
    self.ignored_agents_dir = pathLibrary['ignored_agents_dir']

created on: 5/23/2024
by @LeoBorcherding
"""

import os

# -------------------------------------------------------------------------------------------------
class model_write_class:
    def __init__(self, pathLibrary):
        """a method for initializing the class
        """
        self.pathLibrary = pathLibrary
        self.current_dir = pathLibrary['current_dir']
        self.parent_dir = pathLibrary['parent_dir']
        self.ignored_agents_dir = pathLibrary['ignored_agents_dir']

    # -------------------------------------------------------------------------------------------------    
    def write_model_file(self):
        """ a method to write a model file based on user inputs
            args: none
            returns: none
        """ #TODO ADD WRITE MODEL FILE CLASS
        # collect agent data with text input
        self.user_create_agent_name = input(self.colors['WARNING'] + "<<< PROVIDE SAFETENSOR OR GGUF NAME (WITH .gguf or .safetensors) >>> " + self.colors['OKBLUE'])
        user_input_temperature = input(self.colors['WARNING'] + "<<< PROVIDE NEW AGENT TEMPERATURE (0.1 - 5.0) >>> " + self.colors['OKBLUE'])
        # system_prompt = input(WHITE + "<<< PROVIDE SYSTEM PROMPT >>> " + OKBLUE)

        model_create_dir = os.path.join(self.ignored_agents_dir, f"{self.user_create_agent_name}")
        model_create_file = os.path.join(self.ignored_agents_dir, f"{self.user_create_agent_name}\\modelfile")

        try:
            # Create the new directory
            os.makedirs(model_create_dir, exist_ok=True)
            # Get current model template data
            self.ollama_show_template()
            # Create the text file
            # f.write(f"\n#Set the system prompt\n")
            # f.write(f"SYSTEM \"\"\"\n{system_prompt}\n\"\"\"\n")
            with open(model_create_file, 'w') as f:
                f.write(f"FROM {self.user_create_agent_name}\n")
                f.write(f"#temperature higher -> creative, lower -> coherent\n")
                f.write(f"PARAMETER temperature {user_input_temperature}\n")
                f.write(f"TEMPLATE \"\"\"\n{self.template}\n\"\"\"\n")
            return
        except Exception as e:
            return f"Error creating directory or text file: {str(e)}"
    
    # -------------------------------------------------------------------------------------------------
    def write_model_file_and_run_agent_create_ollama(self):
        """ a method to automatically generate a new agent via commands
            returns: none
        """
        # collect agent data with text input
        self.user_create_agent_name = input(self.colors['WARNING'] + "<<< PROVIDE NEW AGENT NAME TO CREATE >>> " + self.colors['OKBLUE'])
        user_input_temperature = input(self.colors['WARNING'] + "<<< PROVIDE NEW AGENT TEMPERATURE (0.1 - 5.0) >>> " + self.colors['OKBLUE'])
        system_prompt = input(self.colors['WHITE'] + "<<< PROVIDE SYSTEM PROMPT >>> " + self.colors['OKBLUE'])

        model_create_dir = os.path.join(self.ignored_agents_dir, f"{self.user_create_agent_name}")
        model_create_file = os.path.join(self.ignored_agents_dir, f"{self.user_create_agent_name}\\modelfile")

        try:
            # Create the new directory
            os.makedirs(model_create_dir, exist_ok=True)
            # Get current model template data
            self.ollama_show_template()
            # Create the text file
            with open(model_create_file, 'w') as f:
                f.write(f"FROM {self.user_input_model_select}\n")
                f.write(f"#temperature higher -> creative, lower -> coherent\n")
                f.write(f"PARAMETER temperature {user_input_temperature}\n")
                f.write(f"\n#Set the system prompt\n")
                f.write(f"SYSTEM \"\"\"\n{system_prompt}\n\"\"\"\n")
                f.write(f"TEMPLATE \"\"\"\n{self.template}\n\"\"\"\n")

            # Execute create_agent_cmd
            self.create_agent_cmd(self.user_create_agent_name, 'create_agent_automation_ollama.cmd')
            return
        except Exception as e:
            return f"Error creating directory or text file: {str(e)}"

    # -------------------------------------------------------------------------------------------------
    def write_model_file_and_run_agent_create_gguf(self, model_git):
        """ a method to automatically generate a new agent via commands
            args: none
            returns: none
        """
        self.model_git = model_git

        # collect agent data with text input
        self.converted_gguf_model_name = input(self.colors['WARNING'] + "<<< PROVIDE SAFETENSOR OR CONVERTED GGUF NAME (with EXTENTION .gguf or .safetensors) >>> " + self.colors['OKBLUE'])
        self.user_create_agent_name = input(self.colors['WARNING'] + "<<< PROVIDE NEW AGENT NAME TO CREATE >>> " + self.colors['OKBLUE'])
        user_input_temperature = input(self.colors['WARNING'] + "<<< PROVIDE NEW AGENT TEMPERATURE (0.1 - 5.0) >>> " + self.colors['OKBLUE'])
        system_prompt = input(self.colors['WHITE'] + "<<< PROVIDE SYSTEM PROMPT >>> " + self.colors['OKBLUE'])

        model_create_dir = os.path.join(self.ignored_agents_dir, f"{self.user_create_agent_name}")
        model_create_file = os.path.join(self.ignored_agents_dir, f"{self.user_create_agent_name}\\modelfile")
        self.gguf_path_part = os.path.join(self.model_git, "converted")
        self.gguf_path = os.path.join(self.gguf_path_part, f"{self.converted_gguf_model_name}.gguf")
        print(f"model_git: {model_git}")
        try:
            # Create the new directory
            os.makedirs(model_create_dir, exist_ok=True)

            # Copy gguf to IgnoredAgents dir
            self.copy_gguf_to_ignored_agents()

            # Create the text file
            with open(model_create_file, 'w') as f:
                f.write(f"FROM ./{self.converted_gguf_model_name}\n")
                f.write(f"#temperature higher -> creative, lower -> coherent\n")
                f.write(f"PARAMETER temperature {user_input_temperature}\n")
                f.write(f"\n#Set the system prompt\n")
                f.write(f"SYSTEM \"\"\"\n{system_prompt}\n\"\"\"\n")
            
            # Execute create_agent_cmd
            self.create_agent_cmd(self.user_create_agent_name, "create_agent_automation_gguf.cmd")
            return
        except Exception as e:
            return f"Error creating directory or text file: {str(e)}"