class PromptManager:
    def __init__(self, config):
        self.templates = config.prompt_templates
        self.mode = config.model.mode
        
    def build_prompt(self, dialogue, sample_dialogue=None, sample_summary=None, version="1"):
        if self.mode == "prompt":
            template = self.templates[version]
            return self._format_prompt(template, dialogue, sample_dialogue, sample_summary)
        return dialogue
        
    def _format_prompt(self, template, dialogue, sample_dialogue=None, sample_summary=None):
        # 프롬프트 포맷팅 로직
        pass 